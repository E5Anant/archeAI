import requests

def get_weather(location):
  """Fetches and prints weather data including the next day's forecast for the given location.

  Args:
      location (str): The location for which to fetch weather data.
  """
  url = f"https://wttr.in/{location}?format=j1"
  response = requests.get(url)

  if response.status_code == 200:
    data = response.json()
    current = data['current_condition'][0]
    location_name = data['nearest_area'][0]['areaName'][0]['value']

    # Get the forecast for tomorrow 
    tomorrow = data['weather'][1] 
    # date = tomorrow['date']
    max_temp = tomorrow['maxtempC']
    min_temp = tomorrow['mintempC']
    condition = tomorrow['hourly'][4]['weatherDesc'][0]['value']

    return f"Weather in {location_name}: {current['temp_C']}°C, {current['weatherDesc'][0]['value']}, Wind: {current['windspeedKmph']} km/h {current['winddir16Point']}, Humidity: {current['humidity']}%" + f" Forecast for tomorrow: {min_temp}°C - {max_temp}°C, {condition}"

  else:
    print(f"Error: Unable to fetch weather data for {location}. Status code: {response.status_code}")

if __name__ == "__main__":
  print(get_weather("kolkata"))