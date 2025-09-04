import argparse
import logging
import os
import sys
import time
import pathlib

RAW = r"C:\Users\Michi\nurse-stress-wearables\data\merged_data.csv"
OUT = r"C:\Users\Michi\nurse-stress-wearables\data\parquet"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("convert_to_parquet")

def normalize_path(p: str) -> str:
    return p.replace("\\", "/")

def sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def run_duckdb(logger, raw, out, limit=None, subjects=None):
    import duckdb
    raw = normalize_path(raw)
    out = normalize_path(out)

    logger.info("Engine: duckdb")
    logger.info(f"DuckDB version: {duckdb.__version__}")
    con = duckdb.connect()
    try:
        con.execute("SET threads TO 4;")
        con.execute("PRAGMA enable_progress_bar=false;")

        where_subject = ""
        if subjects:
            ids = [s.strip() for s in subjects.split(",") if s.strip()]
            quoted_ids = ", ".join(sql_quote(i) for i in ids)
            where_subject = f"WHERE CAST(id AS VARCHAR) IN ({quoted_ids})"

        limit_clause = f"LIMIT {limit}" if limit else ""

        logger.info("Verifying CSV readability (COUNT(*))...")
        cnt = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{raw}', HEADER=TRUE)"
        ).fetchone()[0]
        logger.info(f"CSV rows reported by DuckDB: {cnt:,}")

        logger.info("Writing partitioned Parquet by id and day (d)...")
        con.execute(
            f"""
            COPY (
                SELECT 
                    CAST(X AS DOUBLE) AS X,
                    CAST(Y AS DOUBLE) AS Y,
                    CAST(Z AS DOUBLE) AS Z,
                    CAST(EDA AS DOUBLE) AS EDA,
                    CAST(HR AS DOUBLE) AS HR,
                    CAST(TEMP AS DOUBLE) AS TEMP,
                    CAST(id AS VARCHAR) AS id,
                    TRY_CAST(datetime AS TIMESTAMP) AS datetime,
                    TRY_CAST(label AS INTEGER) AS label,
                    CAST(date_trunc('day', TRY_CAST(datetime AS TIMESTAMP)) AS DATE) AS d
                FROM read_csv_auto('{raw}', HEADER=TRUE)
                {where_subject}
                {limit_clause}
            )
            TO '{out}'
            (FORMAT PARQUET, PARTITION_BY (id, d), OVERWRITE_OR_IGNORE TRUE);
            """
        )
        logger.info("Parquet write completed, verifying...")
        total_rows = con.execute(
            f"SELECT COUNT(*) FROM parquet_scan('{out}/**/*.parquet')"
        ).fetchone()[0]
        n_subjects = con.execute(
            f"SELECT COUNT(DISTINCT id) FROM parquet_scan('{out}/**/*.parquet')"
        ).fetchone()[0]
        logger.info(f"Verification: rows={total_rows:,}, subjects={n_subjects}")
    finally:
        con.close()
        logger.info("DuckDB connection closed.")

def run_arrow(logger, raw, out, limit=None, subjects=None, chunksize=500_000):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    logger.info("Engine: arrow (pandas + pyarrow)")
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)

    rows_written = 0
    t0 = time.time()

    subject_set = None
    if subjects:
        subject_set = {s.strip() for s in subjects.split(",") if s.strip()}

    for i, chunk in enumerate(pd.read_csv(raw, chunksize=chunksize)):
        # Optional limit by total rows
        if limit is not None and rows_written >= limit:
            break
        logger.info(f"Chunk {i+1}: read {len(chunk):,} rows")

        # Basic typing
        if 'datetime' in chunk.columns:
            chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
            chunk = chunk[chunk['datetime'].notna()]
        chunk['id'] = chunk['id'].astype(str)
        if 'label' in chunk.columns:
            # keep labels nullable; cast errors to NaN then to Int32 nullable
            chunk['label'] = pd.to_numeric(chunk['label'], errors='coerce').astype('Int32')

        # Optional subject filter
        if subject_set is not None:
            chunk = chunk[chunk['id'].isin(subject_set)]

        # Optional limit cap within chunk
        if limit is not None and rows_written + len(chunk) > limit:
            chunk = chunk.iloc[: max(0, limit - rows_written)]

        # Partition column by day
        chunk['d'] = chunk['datetime'].dt.date

        # Convert to Arrow Table
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        # Write partitioned dataset; multiple calls append files
        ds.write_dataset(
            data=table,
            base_dir=out,
            format="parquet",
            partitioning=["id", "d"],
            existing_data_behavior="overwrite_or_ignore",
        )

        rows_written += len(chunk)
        logger.info(f"Chunk {i+1}: wrote {len(chunk):,} rows (total {rows_written:,})")

    logger.info(f"Arrow writer finished in {time.time() - t0:.1f}s, rows_written={rows_written:,}")

def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["duckdb", "arrow"], default="duckdb", help="Conversion engine")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (smoke test)")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject ids to include")
    parser.add_argument("--raw", type=str, default=RAW, help="Path to input CSV")
    parser.add_argument("--out", type=str, default=OUT, help="Output Parquet root")
    args = parser.parse_args()

    raw = args.raw
    out = args.out

    logger.info("Starting CSV -> Parquet conversion")
    logger.info(f"Input: {normalize_path(raw)}")
    logger.info(f"Output dir: {normalize_path(out)}")

    if not os.path.exists(raw):
        logger.error(f"Input file not found: {raw}")
        sys.exit(1)
    try:
        size_mb = os.path.getsize(raw) / (1024**2)
        logger.info(f"Input size: {size_mb:.1f} MB")
    except Exception:
        logger.warning("Could not stat input file size")

    t0 = time.time()
    try:
        if args.engine == "duckdb":
            run_duckdb(logger, raw, out, limit=args.limit, subjects=args.subjects)
        else:
            run_arrow(logger, raw, out, limit=args.limit, subjects=args.subjects)
        logger.info(f"Done in {time.time() - t0:.1f}s")
        print(f"Wrote Parquet partitions to: {OUT}")
    except Exception as e:
        logger.exception(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()