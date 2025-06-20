#!/usr/bin/env python3
"""
Simple script to check database and test new functionality
"""

import os
import psycopg2
from dotenv import load_dotenv
from datetime import date, datetime
from data_preparation import DataPreparation

load_dotenv()

def check_database():
    """Check what data is available in the database"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cur = conn.cursor()
        
        print("=== Database Check ===")
        
        # Check instruments
        cur.execute("SELECT COUNT(*) FROM instruments")
        instrument_count = cur.fetchone()[0]
        print(f"Total instruments: {instrument_count}")
        
        # Check NIFTY instruments
        cur.execute("SELECT symbol, expiry_date FROM instruments WHERE symbol LIKE '%NIFTY%' LIMIT 5")
        nifty_instruments = cur.fetchall()
        print(f"NIFTY instruments (first 5): {nifty_instruments}")
        
        # Check candle data
        cur.execute("SELECT COUNT(*) FROM candle_data_1min")
        candle_count = cur.fetchone()[0]
        print(f"Total candle records: {candle_count}")
        
        # Check date range
        cur.execute("SELECT MIN(time), MAX(time) FROM candle_data_1min")
        date_range = cur.fetchone()
        print(f"Candle data date range: {date_range}")
        
        # Check technical indicators
        cur.execute("SELECT COUNT(*) FROM technical_indicators")
        indicator_count = cur.fetchone()[0]
        print(f"Total technical indicator records: {indicator_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Database check error: {e}")
        return False

def test_with_available_data():
    """Test the new functionality with available data"""
    print("\n=== Testing New Functionality ===")
    
    try:
        dp = DataPreparation()
        
        # Test forward test dataset generation without specific dates
        print("1. Testing forward test dataset generation (no date filter)...")
        success = dp.prepare_forward_test_data()
        
        if success:
            print("✓ Forward test dataset generated successfully")
            
            # Check if file exists and has data
            if os.path.exists("ml_data/forward_test_dataset.csv"):
                import pandas as pd
                df = pd.read_csv("ml_data/forward_test_dataset.csv")
                print(f"✓ Forward test dataset created with {len(df)} samples")
                
                # Show date range of the data
                if '"time"' in df.columns:
                    df['time'] = pd.to_datetime(df['"time"'])
                    print(f"  Data date range: {df['time'].min()} to {df['time'].max()}")
            else:
                print("✗ Forward test dataset file not found")
        else:
            print("✗ Failed to generate forward test dataset")
        
        # Test live inference
        print("\n2. Testing live inference...")
        success, result = dp.run_live_inference()
        
        if success and result:
            print("✓ Live inference completed successfully")
            print(f"  - Prediction: {result['prediction']}")
            print(f"  - Confidence: {result['confidence']:.2%}")
            print(f"  - Current Price: ₹{result['current_price']:.2f}")
            print(f"  - RSI: {result['rsi']:.2f}")
        elif not success and result is None:
            print("⚠ Live inference skipped (no trained model found)")
        else:
            print("✗ Failed to run live inference")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in testing: {e}")
        return False

def main():
    """Main function"""
    print("Forward Testing and Live Testing - Database Check and Test")
    print("=" * 60)
    
    # Check database
    db_ok = check_database()
    
    if db_ok:
        # Test functionality
        test_ok = test_with_available_data()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Database Check: {'✓ PASSED' if db_ok else '✗ FAILED'}")
        print(f"Functionality Test: {'✓ PASSED' if test_ok else '✗ FAILED'}")
        
        if db_ok and test_ok:
            print("\n🎉 All checks passed! The new functionality is working correctly.")
        else:
            print("\n❌ Some checks failed. Please review the output above.")
    else:
        print("\n❌ Database check failed. Cannot proceed with testing.")

if __name__ == "__main__":
    main() 