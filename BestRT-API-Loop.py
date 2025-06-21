#from socketclusterclient import Socketcluster
#mport Socketcluster  #using modified socketcluster in current dir, instead of the one installed pip(file SocketCluster.py in current dir)
import logging
import requests 
import pandas as pd
from datetime import datetime
import os
import time
        
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        
        
def onconnect(socket):
    logging.info("on connect got called")


def ondisconnect(socket):
    logging.info("on disconnect got called")


def onConnectError(socket, error):
    logging.info("On connect error got called")


def onSetAuthentication(socket, token):
    logging.info("Token received " + token)
    socket.setAuthtoken(token)


def onAuthentication(socket, isauthenticated):
    logging.info("Authenticated is " + str(isauthenticated))
    #subscribe_to_channel(socket,'yell')
    subscribe_to_first_five_tickers(socket)

def suback(channel, error, object):
    if error == '':
        print ("Subscribed successfully to channel " , channel)

def unsuback(channel, error, object):
    if error == '':
       print ("Unsubscribed to channel " , channel)

def puback(channel, error, object):
    if error == '':
        logging.info("Publish sent successfully to channel " + channel)

def ack(key, error, object):
    logging.info("Got ack data " + object + " and error " + error + " and key is " + key)

def message(key, object):
    logging.info("Got data " + object + " from key " + key)


def messsageack(key, object, ackmessage):
    logging.info("Got data " + object + " from key " + key)
    ackmessage("this is error", "this is data")

def pong(key, object, ackmessage):
    logging.info("Got data " + object + " from key " + key)
    ackmessage("", "#2")

        

def channelmessage(key, object):
    logging.info("Got channel data " + object + " from key " + key)


#############################################################################################


def get_auth_token(loginId, product, apikey):
    authEndPoint = "http://s3.vbiz.in/directrt/gettoken?loginid={}&product={}&apikey={}".format(loginId, product, apikey)
    try:
        res = requests.get(authEndPoint)
        if res.status_code == 200:
            return True, 'Access token returned successfully', res.json()['AccessToken']
        else:
            return False, 'Response code returned not 200', res.text
    except Exception as e:
        return False, 'Network error occurred trying to get access token', str(e)

def subscribe_to_channel(socket,channel):
    # without acknowledgement
    #socket.subscribe('yell')
    
    #with acknowledgement
    #socket.subscribeack('yell', suback) #suback is the function you want  to be called,when acknowledgement is received from server
    socket.subscribeack(channel, suback)


def unsubscribe_from_channel(socket,channel):
    # without acknowledgement
    #socket.unsubscribe('yell')
    
    # with acknowledgement
    #socket.unsubscribeack('yell', unsuback) #unsuback is the function you want  to be called,when acknowledgement is received from server
    socket.unsubscribeack(channel, unsuback)


def get_all_subscribed_channels(socket):
    channels = socket.getsubscribedchannels()
    print('all subscribed_channels: ',channels)
    return True,channels

def download_tickers(loginId,product,access_token):
    try:
        #gettickers = "gettickers?loginid={loginid}&product={product}&accesstoken={accesstkn}"
        reqEndPoint = "http://qbase1.vbiz.in/directrt/"
        #url = reqEndPoint + addurlqueries(gettickers)
        url = "http://qbase1.vbiz.in/directrt/gettickers?loginid={}&product={}&accesstoken={}".format(loginId,product,access_token)
        res = requests.get(url)
        if not res.status_code == 200:
            #error occured getting the tickers
            return False, 'Error occured getting tickers',res.status_code
        else:
            #write the tickers to file(overwrite)
            with open('tickers.txt','w') as f:
                f.write(res.text)
                print('tickers downloaded, saved to file `tickers.txt`')
    except Exception as e:
        print('Exception occured while trying to get tickers: ',str(e))
        return False,'Exception occured while trying to get tickers',str(e)

def subscribe_to_first_five_tickers(socket):
    nse_tickers = ['NSE_OPTIDX_NIFTY_05JUN2025_24700_CE.csv',
'NSE_OPTIDX_NIFTY_05JUN2025_24700_CE.json']


    for ticker in nse_tickers:
        subscribe_to_channel(socket,ticker)
        socket.onchannel(ticker, channelmessage)


def get_historical_data(
    loginId, product, access_token, startdate, enddate,
    exch, inst, symbol, expiry=None, strike=None, optiontype=None, delay=0.6
):
    """
    Fetch historical data from the API.
    Dates should be in format: DDMMMYYYY (e.g., 12FEB2025)
    For options, provide expiry, strike, and optiontype.
    delay: Delay in seconds between API calls to respect rate limits (default 0.6s = max 1.67 calls/sec)
    """
    url = (
        "http://qbase1.vbiz.in/directrt/gethistorical"
        "?loginid={}&product={}&accesstoken={}&startdate={}&enddate={}&exch={}&inst={}&symbol={}"
    ).format(loginId, product, access_token, startdate, enddate, exch, inst, symbol)
    if expiry:
        url += f"&expiry={expiry}"
    if strike:
        url += f"&strike={strike}"
    if optiontype:
        url += f"&optiontype={optiontype}"
    
    # Add delay to respect rate limits (2 requests per second max)
    if delay > 0:
        time.sleep(delay)
    
    try:
        res = requests.get(url, timeout=30)  # Added timeout
        if res.status_code == 200:
            try:
                # Process the response as CSV
                from io import StringIO
                csv_data = StringIO(res.text)
                df = pd.read_csv(csv_data)
                return True, df
            except Exception as e:
                # If CSV parsing fails, return raw text for inspection
                return False, f"CSV parsing error: {str(e)} | Raw response: {res.text[:200]}..."
        elif res.status_code == 429:  # Rate limit error
            return False, "RATE_LIMIT_ERROR: Too many requests"
        else:
            return False, f"HTTP {res.status_code}: {res.text[:200]}..."
    except requests.exceptions.Timeout:
        return False, "TIMEOUT_ERROR: Request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"REQUEST_ERROR: {str(e)}"
    except Exception as e:
        return False, f"UNEXPECTED_ERROR: {str(e)}"

def convert_date_format(date_str):
    """
    Convert date from DD-MMM-YY or DD-MMM-YYYY or DDMMMYYYY format to DDMMMYYYY format
    Example: "22-Dec-18" -> "22DEC2018", "03Jan2019" -> "03JAN2019"
    """
    try:
        # If already in DDMMMYYYY format, just return uppercase
        if len(date_str) == 9 and date_str[2:5].isalpha():
            return date_str.upper()
        
        # Handle different separators and formats
        if '-' in date_str:
            parts = date_str.split('-')
            if len(parts) == 3:
                day, month, year = parts
                # Handle 2-digit year
                if len(year) == 2:
                    parsed_date = datetime.strptime(date_str, "%d-%b-%y")
                else:  # 4-digit year
                    parsed_date = datetime.strptime(date_str, "%d-%b-%Y")
                return parsed_date.strftime("%d%b%Y").upper()
        
        # Try parsing as DDMMMYYYY or other formats
        try:
            parsed_date = datetime.strptime(date_str, "%d%b%Y")
            return parsed_date.strftime("%d%b%Y").upper()
        except:
            # Try other common formats
            for fmt in ["%d%b%y", "%d-%b-%Y", "%d-%b-%y"]:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime("%d%b%Y").upper()
                except:
                    continue
        
        # If all parsing fails, return original string
        print(f"Warning: Could not parse date format for {date_str}, using as-is")
        return date_str.upper()
        
    except Exception as e:
        print(f"Error converting date {date_str}: {e}")
        return date_str.upper()

def process_nifty_input_file(loginId, product, access_token, input_file="Nifty_Input.csv", max_rows=None, api_delay=0.6):
    """
    Process the Nifty input CSV file and fetch historical data for all combinations
    api_delay: Delay between API calls in seconds (default 0.6 = ~1.67 calls/sec, under 2/sec limit)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Limit rows if specified
        if max_rows:
            df = df.head(max_rows)
            print(f"Processing first {len(df)} rows from {input_file}")
        else:
            print(f"Processing all {len(df)} rows from {input_file}")
        
        print(f"API delay set to {api_delay} seconds between calls (max {1/api_delay:.1f} calls/sec)")
        
        total_combinations = 0
        successful_downloads = 0
        failed_combinations = []
        rate_limit_errors = 0
        
        for index, row in df.iterrows():
            symbol = row['Symbol']
            start_date = convert_date_format(row['StartDate'])
            end_date = convert_date_format(row['EndDate'])
            expiry = convert_date_format(row['Expiry'])
            low_strike = int(row['Low'])
            high_strike = int(row['High'])
            
            print(f"\nProcessing row {index + 1}/{len(df)}")
            print(f"Symbol: {symbol}, Period: {start_date} to {end_date}")
            print(f"Expiry: {expiry}, Strike range: {low_strike} to {high_strike}")
            
            # Loop through strikes in increments of 50
            strikes = range(low_strike, high_strike + 1, 50)
            option_types = ['CE', 'PE']
            
            row_successful = 0
            row_total = len(strikes) * len(option_types)
            
            print(f"  Will make {row_total} API calls for this row...")
            
            for strike_idx, strike in enumerate(strikes):
                for opt_idx, option_type in enumerate(option_types):
                    total_combinations += 1
                    
                    # Create dynamic filename: NIFTY26JUN202556300CE.csv
                    filename = f"{symbol}{expiry}{strike}{option_type}.csv"
                    
                    # Show progress within the row
                    current_call = strike_idx * len(option_types) + opt_idx + 1
                    print(f"    [{current_call:2d}/{row_total:2d}] {symbol} {expiry} {strike} {option_type}...", end=" ")
                    
                    # Fetch historical data with rate limiting
                    success, data = get_historical_data(
                        loginId, product, access_token,
                        start_date, end_date, "NSE", "OPTIDX",
                        symbol, expiry, str(strike), option_type, api_delay
                    )
                    
                    if success:
                        try:
                            # Check if data is not empty
                            if len(data) > 0:
                                # Save to CSV file
                                data.to_csv(filename, index=False)
                                print(f"✓ ({len(data)} rows)")
                                successful_downloads += 1
                                row_successful += 1
                            else:
                                print("✓ (No data)")
                                successful_downloads += 1
                                row_successful += 1
                        except Exception as e:
                            print(f"✗ Save error: {e}")
                            failed_combinations.append(f"{filename}: Save error - {e}")
                    else:
                        # Handle different error types
                        if "RATE_LIMIT_ERROR" in str(data):
                            rate_limit_errors += 1
                            print(f"✗ Rate limit")
                            failed_combinations.append(f"{filename}: Rate limit exceeded")
                            # Extra delay after rate limit error
                            print(f"    Waiting extra 2 seconds due to rate limit...")
                            time.sleep(2)
                        elif "TIMEOUT_ERROR" in str(data):
                            print(f"✗ Timeout")
                            failed_combinations.append(f"{filename}: Request timeout")
                        elif "No data" in str(data) or len(str(data)) < 50:
                            print(f"✗ No data")
                            failed_combinations.append(f"{filename}: No data available")
                        else:
                            print(f"✗ API error")
                            failed_combinations.append(f"{filename}: {str(data)[:100]}...")
            
            print(f"  Row summary: {row_successful}/{row_total} combinations successful")
            
            # No additional delay between rows since we already have delays between calls
        
        print(f"\n=== Final Summary ===")
        print(f"Total combinations processed: {total_combinations}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {total_combinations - successful_downloads}")
        print(f"Rate limit errors: {rate_limit_errors}")
        
        if failed_combinations:
            print(f"\nFirst 10 failures:")
            for i, failure in enumerate(failed_combinations[:10]):
                print(f"  {i+1}. {failure}")
            if len(failed_combinations) > 10:
                print(f"  ... and {len(failed_combinations) - 10} more")
                
        if rate_limit_errors > 0:
            print(f"\n⚠️  Warning: {rate_limit_errors} rate limit errors detected!")
            print(f"   Consider increasing api_delay (currently {api_delay}s)")
            print(f"   Recommended: Try {api_delay + 0.2}s or higher")
        
    except Exception as e:
        print(f"Error processing input file: {e}")
        return False
    
    return True

def main():
    loginId = 'DC-BPRA9112'  # Replace with your login ID
    product = 'DIRECTRTLITE'  # Replace with your product
    apikey = 'B4D2ECAFA32941AC93ED'  # Replace with your API key

    # Get access token
    res = get_auth_token(loginId, product, apikey)
    if not res[0]:
        print(res[1], res[2])
        return

    # Access token returned successfully, if execution reaches here.
    access_token = res[2]
    print('Access token obtained successfully')

    # Check if input file exists
    input_file = "Nifty_Input.csv"
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please ensure the Nifty_Input.csv file is in the same directory as this script.")
        return    # Ask user how many rows to process (for testing)
    try:
        max_rows = input("Enter number of rows to process (press Enter for all rows): ").strip()
        if max_rows:
            max_rows = int(max_rows)
            print(f"Processing first {max_rows} rows only...")
        else:
            max_rows = None
            print("Processing all rows...")
    except ValueError:
        print("Invalid input. Processing all rows...")
        max_rows = None

    # Ask user for API delay setting
    try:
        delay_input = input("Enter API delay in seconds (press Enter for default 0.6s): ").strip()
        if delay_input:
            api_delay = float(delay_input)
            if api_delay < 0.5:
                print("⚠️  Warning: Delay less than 0.5s may exceed rate limit (2 requests/sec)")
            print(f"API delay set to {api_delay} seconds...")
        else:
            api_delay = 0.6
            print("Using default API delay of 0.6 seconds...")
    except ValueError:
        print("Invalid input. Using default API delay of 0.6 seconds...")
        api_delay = 0.6

    # Process the input file and fetch all historical data
    print(f"Starting to process {input_file}...")
    success = process_nifty_input_file(loginId, product, access_token, input_file, max_rows, api_delay)
    
    if success:
        print("✓ Processing completed successfully!")
    else:
        print("✗ Processing failed!")

    # Original single example (commented out)
    # success, data = get_historical_data(
    #     loginId, product, access_token,
    #     "21NOV2023", "23NOV2023", "NSE", "OPTIDX",
    #     "NIFTY", "23NOV2023", "19700", "CE"
    # )
    # if success:
    #     # Export DataFrame to CSV
    #     output_file = input("Enter the output CSV file name (default: historical_data.csv): ").strip()
    #     if not output_file:
    #         output_file = "historical_data.csv"
    #
    #     try:
    #         data.to_csv(output_file, index=False)
    #         print(f"Historical data exported to {output_file}")
    #         print(data.describe(include='all'))
    #         print("\nFirst row:")
    #         print(data.iloc[0])
    #         print("\nLast row:")
    #         print(data.iloc[-1])
    #     except Exception as e:
    #         print(f"Error exporting data to {output_file}: {e}")
    # else:
    #     print("Failed to fetch historical data:", data)

    # Connect to websocket
    wsEndPoint = "ws://116.202.165.216:992/directrt/?loginid={}&accesstoken={}&product={}".format(loginId, access_token, product)
    # print('final websocket endpoint to call: ', wsEndPoint)
    # socket = Socketcluster.socket(wsEndPoint)
    # socket.setBasicListener(onconnect, ondisconnect, onConnectError)
    # socket.setAuthenticationListener(onSetAuthentication, onAuthentication)
    # socket.onack('ping', pong)  # listening to event
    # socket.on('unsubscribed', message)  # listening to event
    # socket.on('subscribed', message)  # listening to event
    # socket.onchannel('yell', channelmessage)  # listening to channel
    # socket.enablelogger(True)
    # socket.connect()


if __name__ == '__main__':
    main()