from typing import List

from model.StatelineParser import StatelineParser
from model.model_root import ModelRoot

class WeatherForecastPrediction(ModelRoot):
    def __init__(self,
                 time          = 0,
                 temperature   = 0.0,
                 windSpeed     = 0.0,
                 windDirection = 0.0,
                 cloudCover    = 0.0,
                 origin        = 0):

        self.forecastTime  = time
        self.temperature   = temperature
        self.windSpeed     = windSpeed
        self.windDirection = windDirection % 360
        self.cloudCover    = cloudCover
        self.origin        = origin

    @staticmethod
    def from_state_line(line: str)-> "WeatherForecastPrediction":
        parts = StatelineParser.split_line(line)

        return WeatherForecastPrediction(time          = int(parts[3]),
                                         temperature   = float(parts[4]),
                                         windSpeed     = float(parts[5]),
                                         windDirection = float(parts[6]),
                                         cloudCover    = float(parts[7]))


class WeatherForecast(object):

    """holds a set of forecast predictions (24)"""

    def __init__(self,
                 currentTimeslot,
                 predictions: List[WeatherForecastPrediction]
                 ):
        self.currentTimeslot = currentTimeslot
        self.predictions     = predictions


class WeatherReport:
    def __init__(self,
                 currentTimeslot      = 0,
                 temperature   = 0.0,
                 windSpeed     = 0.0,
                 windDirection = 0.0,
                 cloudCover    = 0.0):

        self.currentTimeslot = currentTimeslot
        self.temperature     = temperature
        self.windSpeed       = windSpeed
        self.windDirection   = windDirection % 360
        self.cloudCover      = cloudCover

    @staticmethod
    def from_state_line(line: str)-> "WeatherReport":
        parts = StatelineParser.split_line(line)

        return WeatherReport (currentTimeslot= int(parts[3]),
                              temperature   = float(parts[4]),
                              windSpeed     = float(parts[5]),
                              windDirection = float(parts[6]),
                              cloudCover    = float(parts[7]))
