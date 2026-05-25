import matplotlib.pyplot as plt
import pandas as pd
import os

def export_table_image(df: pd.DataFrame, title: str, output_path: str, num_rows: int = 10):
    """
    Render a pandas DataFrame as a beautiful table image using Matplotlib.
    """
    # Take a sample of the data
    sample_df = df.head(num_rows).copy()
    
    # Format numbers for beauty
    for col in sample_df.columns:
        if sample_df[col].dtype == 'float64':
            sample_df[col] = sample_df[col].map('{:,.2f}'.format)
        if sample_df[col].dtype == 'int64':
            sample_df[col] = sample_df[col].map('{:,}'.format)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#1f77b4')

    # Create the table
    table = ax.table(
        cellText=sample_df.values,
        colLabels=sample_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#1f77b4'] * len(sample_df.columns)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Header style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('white')
            cell.set_facecolor('#1f77b4')
        else:
            # Alternating row colors
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('white')

    # Ensure reports directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Exported beautiful table to: {output_path}")
    
    # Optional: Pop up the window (uncomment if you want it to show every time)
    # plt.show() 
    
    plt.close()
