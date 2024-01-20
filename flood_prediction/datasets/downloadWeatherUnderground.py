import requests
from datetime import datetime, date
import argparse
import os

# Function to perform API requests
def get_weather_data(station_id):
    _date =  date.today().strftime('%Y%m%d')
    const api_key = 
    url = f"https://api.weather.com/v2/pws/history/all?stationId={station_id}&format=json&units=e&date={date}&apiKey={api_key}"
    response = requests.get(url)
    return response.json()

# # Parse command line arguments
# parser = argparse.ArgumentParser(description='Download weather data.')
# parser.add_argument('-a', '--apikey', required=True, help='API key for weatherunderground')
# parser.add_argument('-s', '--stationid', default='KLALAFAY248', help='Station ID')
# parser.add_argument('-d', '--date', default=date.today().strftime('%Y%m%d'), help='Start date in YYYYMMDD format')
# parser.add_argument('-f', '--filename', help='Output filename')
# args = parser.parse_args()

# Check for required `wget` command (optional in Python)
# if not shutil.which('wget'):
#     print("Error: wget is not installed. Please install and re-run", file=sys.stderr)
#     sys.exit(1)

# Date manipulation
start_date = datetime.datetime.strptime(_date, '%Y%m%d')
end_date = start_date - datetime.timedelta(days=180)
current_date = start_date

# Output file setup
output_dir = '~/Documents/2023_programs/weather/water_level_prediction/datasets/KLALAFAY'
output_file = os.path.join(output_dir, f"{args.stationid}-{args.date}.txt")
output_file = os.path.expanduser(output_file)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Data download loop
no_data_count = 0
while current_date >= end_date and no_data_count < 30:
    data = get_weather_data(args.apikey, args.stationid, current_date.strftime('%Y%m%d'))
    if data.get("observations") == []:
        no_data_count += 1
    else:
        no_data_count = 0
        with open(output_file, 'a') as file:
            file.write(str(data) + '\n')
    current_date -= datetime.timedelta(days=1)

print(f"Data download completed. Data saved to {output_file}")