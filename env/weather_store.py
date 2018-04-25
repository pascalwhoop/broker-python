from ast import literal_eval

import util.config as cfg
import logging

from model.StatelineParser import StatelineParser
from model.weather import WeatherForecastPrediction, WeatherReport


class WeatherStore():
    def __init__(self, env):
        self.env = env
        self.weather_forecasts   = {}
        self.weather_predictions = {} #keys are origin+FC --> "360+14" --> Obj
        self.weather_reports     = {}


    def handle_weatherForecastPrediction_new(self, line: str):
        parts       = StatelineParser.split_line(line)
        key = parts[1]
        prediction = WeatherForecastPrediction.from_state_line(line)
        prediction.origin = self.env.current_timestep
        self.weather_predictions[key] = prediction

    def handle_weatherForecast_new(self, line:str):
        #6651:org.powertac.common.WeatherForecast::602::new::360::(26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49)
        parts       = StatelineParser.split_line(line)
        origin = parts[-2]
        ids    = parts[-1]
        ids = literal_eval(ids)
        for id in ids:
            key = str(id)
            if key in self.weather_predictions:
                prediction = self.weather_predictions.pop(key)
                self.weather_predictions["{}+{}".format(origin,prediction.forecastTime)] = prediction
            else:
                logging.warning("weather not found {} -- {} -- {}".format(key, origin, id))


    def handle_weatherReport_new(self, line: str):
        report = WeatherReport.from_state_line(line)
        self.weather_reports[report.currentTimeslot] = report
        #TODO keep reports
        return None

