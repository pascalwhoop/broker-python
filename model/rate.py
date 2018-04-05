from datetime import datetime
from util import config as cfg
from model.model_root import ModelRoot


class Rate(ModelRoot):
    """
    Holds a rate from powertac tariffs.
    * State log fields for readResolve():<br>
 * <code>new(long tariffId,  weeklyBegin,  weeklyEnd,<br>
 *    dailyBegin,  dailyEnd, double tierThreshold,<br>
 *   boolean fixed, double minValue, double maxValue,<br>
 *   long noticeInterval, double expectedMean, double maxCurtailment)</code>
    """

    def __init__(self,
                 id_,
                 tariffId       = None,
                 weeklyBegin    = -1,
                 weeklyEnd      = -1,
                 dailyBegin     = -1,
                 dailyEnd       = -1,
                 tierThreshold  = 0.0,
                 fixed          = True,
                 minValue       = 0.0,
                 maxValue       = 0.0,
                 noticeInterval = 0,
                 expectedMean   = 0.0,
                 maxCurtailment = 0.0):

        self.id_            = id_
        self.tariffId       = tariffId
        self.weeklyBegin    = weeklyBegin
        self.weeklyEnd      = weeklyEnd
        self.dailyBegin     = dailyBegin
        self.dailyEnd       = dailyEnd
        self.tierThreshold  = tierThreshold
        self.fixed          = fixed
        self.minValue       = minValue
        self.maxValue       = maxValue
        self.noticeInterval = noticeInterval
        self.expectedMean   = expectedMean
        self.maxCurtailment = maxCurtailment

    @staticmethod
    def from_state_line(line, ):
        parts = line.split("::")

        id_            =             parts[1]
        tariffId       =             parts[3]
        weeklyBegin    =         int(parts[4])
        weeklyEnd      =         int(parts[5])
        dailyBegin     =         int(parts[6])
        dailyEnd       =         int(parts[7])
        tierThreshold  = round(float(parts[8]),  cfg.ROUNDING_PRECISION)
        fixed          =             parts[9] == 'true'  # boolean value
        minValue       = round(float(parts[10]), cfg.ROUNDING_PRECISION)
        maxValue       = round(float(parts[11]), cfg.ROUNDING_PRECISION)
        noticeInterval =         int(parts[12])
        expectedMean   = round(float(parts[13]), cfg.ROUNDING_PRECISION)
        maxCurtailment = round(float(parts[14]), cfg.ROUNDING_PRECISION)

        return Rate(id_,
                    tariffId,
                    weeklyBegin,
                    weeklyEnd,
                    dailyBegin,
                    dailyEnd,
                    tierThreshold,
                    fixed,
                    minValue,
                    maxValue,
                    noticeInterval,
                    expectedMean,
                    maxCurtailment)

    def is_applicable(self, d: datetime) -> bool:
        """Code taken from PowerTAC server"""
        applies_weekly = False
        applies_daily = False

        # // check # weekly # applicability
        if self.weeklyBegin is -1 or self.weeklyEnd is -1:
            applies_weekly = True
        elif self.weeklyEnd >= self.weeklyBegin:
            applies_weekly = (self.weeklyBegin <= d.isoweekday() <= self.weeklyEnd)
        else:
            applies_weekly = d.isoweekday() >= self.weeklyBegin or d.isoweekday() <= self.weeklyEnd


        # // check # daily # applicability
        if self.dailyBegin is -1 or self.dailyEnd is -1:
            applies_daily = True
        elif self.dailyEnd > self.dailyBegin:
            applies_daily = self.dailyBegin <= d.hour <= self.dailyEnd
        else:
           applies_daily = d.hour >= self.dailyBegin or d.hour <= self.dailyEnd

        return applies_daily and applies_weekly
