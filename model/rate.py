import Config as cfg


class Rate:
    """
    * State log fields for readResolve():<br>
 * <code>new(long tariffId,  weeklyBegin,  weeklyEnd,<br>
 * &nbsp;&nbsp; dailyBegin,  dailyEnd, double tierThreshold,<br>
 * &nbsp;&nbsp;boolean fixed, double minValue, double maxValue,<br>
 * &nbsp;&nbsp;long noticeInterval, double expectedMean, double maxCurtailment)</code>
    """

    def __init__(self, tariff_id, weekly_begin, weekly_end, daily_begin, daily_end, tier_threshold, fixed, min_value,
                 max_value, notice_interval, expected_mean, max_curtailment):
        self.tariff_id = tariff_id
        self.weekly_begin = weekly_begin
        self.weekly_end = weekly_end
        self.daily_begin = daily_begin
        self.daily_end = daily_end
        self.tier_threshold = tier_threshold
        self.fixed = fixed
        self.min_value = min_value
        self.max_value = max_value
        self.notice_interval = notice_interval
        self.expected_mean = expected_mean
        self.max_curtailment = max_curtailment

    @staticmethod
    def from_state_line(line, ):
        parts = line.split("::")

        _tariff_id = parts[3]
        _weekly_begin = int(parts[4])
        _weekly_end = int(parts[5])
        _daily_begin = int(parts[6])
        _daily_end = int(parts[7])
        _tier_threshold = round(float(parts[8]), cfg.ROUNDING_PRECISION)
        _fixed = parts[9] == 'true'  # boolean value
        _min_value = round(float(parts[10]), cfg.ROUNDING_PRECISION)
        _max_value = round(float(parts[11]), cfg.ROUNDING_PRECISION)
        _notice_interval = int(parts[12])
        _expected_mean = round(float(parts[13]), cfg.ROUNDING_PRECISION)
        _max_curtailment = round(float(parts[14]), cfg.ROUNDING_PRECISION)

        return Rate(_tariff_id, _weekly_begin, _weekly_end, _daily_begin, _daily_end, _tier_threshold, _fixed,
                    _min_value, _max_value, _notice_interval, _expected_mean, _max_curtailment)
