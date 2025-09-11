import pandas as pd

def get_standardized_non_derivatives_mapping():
    """Return a mapping of original column names to standardized names for non-derivatives data."""
    return {
        'SYMBOL': 'symbol',
        'COMPANY': 'company_name',
        'REGULATION': 'regulation',
        'NAME OF THE ACQUIRER/DISPOSER': 'person_name',
        'CATEGORY OF PERSON': 'person_category',
        'TYPE OF SECURITY (PRIOR)': 'security_type_prior',
        'NO. OF SECURITY (PRIOR)': 'shares_prior',
        '% SHAREHOLDING (PRIOR)': 'shareholding_pct_prior',
        'NO. OF SECURITIES (ACQUIRED/DISPLOSED)': 'shares_traded',
        'VALUE OF SECURITY (ACQUIRED/DISPLOSED)': 'value_traded',
        'ACQUISITION/DISPOSAL TRANSACTION TYPE': 'transaction_type',
        'TYPE OF SECURITY (POST)': 'security_type_post',
        'NO. OF SECURITY (POST)': 'shares_post',
        '% POST': 'shareholding_pct_post',
        'DATE OF ALLOTMENT/ACQUISITION FROM': 'date_from',
        'DATE OF ALLOTMENT/ACQUISITION TO': 'date_to',
        'DATE OF INITMATION TO COMPANY': 'date_intimation',
        'MODE OF ACQUISITION': 'transaction_mode',
        'EXCHANGE': 'exchange',
        'BROADCASTE DATE AND TIME': 'broadcast_datetime',
        'XBRL': 'xbrl_url'
    }

def get_standardized_derivatives_mapping():
    """Return a mapping of original column names to standardized names for derivatives data."""
    return {
        'SYMBOL': 'symbol',
        'COMPANY': 'company_name',
        'REGULATION': 'regulation',
        'NAME OF THE ACQUIRER/DISPOSER': 'person_name',
        'CATEGORY OF PERSON': 'person_category',
        'DERIVATIVE TYPE SECURITY': 'derivative_type',
        'DERIVATIVE CONTRACT SPECIFICATION': 'contract_spec',
        'NOTIONAL VALUE(BUY)': 'notional_value_buy',
        'NUMBER OF UNITS/CONTRACT LOT SIZE (BUY)': 'units_buy',
        'NOTIONAL VALUE(SELL)': 'notional_value_sell',
        'NUMBER OF UNITS/CONTRACT LOT SIZE  (SELL)': 'units_sell',
        'EXCHANGE': 'exchange',
        'BROADCASTE DATE AND TIME': 'broadcast_datetime',
        'XBRL': 'xbrl_url'
    }

def clean_column_name(col):
    """Clean a column name by removing newlines, extra spaces, and standardizing format."""
    # Remove newlines and extra spaces
    col = col.replace('\n', '').strip()
    # Remove trailing spaces and \n
    col = col.rstrip(' \n')
    return col

def standardize_columns():
    print("Loading files...")
    # Read both CSV files
    try:
        non_derivatives = pd.read_csv('nse_cleaned_final.csv')
        derivatives = pd.read_csv('nse_derivatives_cleaned.csv')
        
        print("\nOriginal non-derivatives columns:")
        for col in non_derivatives.columns:
            print(f"  {col}")
            
        print("\nOriginal derivatives columns:")
        for col in derivatives.columns:
            print(f"  {col}")
        
        # Clean column names
        non_derivatives.columns = [clean_column_name(col) for col in non_derivatives.columns]
        derivatives.columns = [clean_column_name(col) for col in derivatives.columns]
        
        # Get the standardized column mappings
        non_derivatives_mapping = get_standardized_non_derivatives_mapping()
        derivatives_mapping = get_standardized_derivatives_mapping()
        
        # Rename columns using the mappings
        non_derivatives = non_derivatives.rename(columns=non_derivatives_mapping)
        derivatives = derivatives.rename(columns=derivatives_mapping)
        
        print("\nStandardized non-derivatives columns:")
        for col in non_derivatives.columns:
            print(f"  {col}")
            
        print("\nStandardized derivatives columns:")
        for col in derivatives.columns:
            print(f"  {col}")
        
        # Save the files with standardized column names
        print("\nSaving files with standardized column names...")
        non_derivatives.to_csv('nse_non_derivatives_cleaned_standardized.csv', index=False)
        derivatives.to_csv('nse_derivatives_cleaned_standardized.csv', index=False)
        print("Files saved successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the input files - {str(e)}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")

if __name__ == "__main__":
    standardize_columns() 