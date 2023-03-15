import datetime
from datetime import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from finalProject.monthly_reccomendation import hotel_sd_mapping, booking_maps, hotel_gt_booking


def monthTrendyPlot(hotels_count):
    '''plot # of trendy hotels based on month '''
    assert isinstance(hotels_count, list)
    assert len(hotels_count) == 12
    for i in hotels_count:
        assert isinstance(i, int)

    forDrawingData = []
    for i in range(1, 13):
        for j in range(hotels_count[i - 1]):
            forDrawingData.append(i)
    # print(forDrawingData)
    fig, ax = plt.subplots()
    bins = np.arange(1, 14)
    values, xxx, bars = ax.hist(forDrawingData, bins=bins, edgecolor="k", align='left', rwidth=0.8)
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([datetime.date(2020, i, 1).strftime('%b') for i in bins[:-1]])
    plt.ylabel("# of trendy hotel")
    ax.set_title('Monthly recommendation')
    plt.bar_label(bars, fontsize=12, color='navy')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("intermediates/df.csv")
    df2 = df[['hotelID', 'BookDate']]
    month_map = booking_maps(df2)
    hotel_list = hotel_gt_booking(df2, 250)
    hotels_count = []
    for month in range(1, 12 + 1):
        mapping = hotel_sd_mapping(month_map, hotel_list, month, weightEqual=False, sd_diff=1.5)
        hotels_count.append(len(mapping))
    monthTrendyPlot(hotels_count)