import pandas as pd

def create_subset(input_path, output_path, n_subjects=3):
    """Create a subset of the dataset with diverse stress patterns."""
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Calculate stress percentages per subject
    subject_stats = df.groupby('id')['label'].agg(['count', lambda x: (x != 0).sum()])
    subject_stats.columns = ['total_records', 'stress_records']
    subject_stats['stress_percentage'] = (subject_stats['stress_records'] / subject_stats['total_records'] * 100)
    subject_stats = subject_stats.sort_values('stress_percentage', ascending=False)
    
    # Select diverse subjects
    n_total = len(subject_stats)
    if n_subjects == 3:
        indices = [0, n_total // 2, n_total - 1]  # high, medium, low
    else:
        indices = [i * (n_total - 1) // (n_subjects - 1) for i in range(n_subjects)]
    
    selected_subjects = subject_stats.iloc[indices].index.tolist()
    
    # Create and save subset
    subset_df = df[df['id'].isin(selected_subjects)].copy()
    subset_df['datetime'] = pd.to_datetime(subset_df['datetime'])
    subset_df = subset_df.sort_values(['id', 'datetime']).reset_index(drop=True)
    subset_df.to_csv(output_path, index=False)
    
    # Summary
    print(f"Selected {len(selected_subjects)} subjects: {selected_subjects}")
    print(f"Dataset reduced from {len(df):,} to {len(subset_df):,} rows ({len(subset_df)/len(df)*100:.1f}%)")
    
    return subset_df

if __name__ == "__main__":
    input_path = r'C:\Users\Michi\nurse-stress-wearables\data\merged_data.csv'
    output_path = r'C:\Users\Michi\nurse-stress-wearables\data\merged_data_subset.csv'
    
    subset_df = create_subset(input_path, output_path, n_subjects=3)