# ECE143_Hotel_Recommendation

This project was completed by Group 8 for the Programming and Data Analysis(ECE143) coursework. The objective of this project is to create a hotel recommendation system which incorporates hotel trends along with general recommendation system and result analysis based on the data shared by local hotel booking website. 

# Main Project Files

* ```src/main.py``` :runs the complete pipeline, invoking collaborative filtering based recommendation as well as trend based recommendation
* ```src/app.py``` :provides a workable GUI which allows the user to play around with different weights corresponding to Trend and Collaborative Filtering recommendations.
* ```eda_jupyter_notebooks/data_analysis.ipynb``` :juyter notebook which shows all the visualisations

# Usage Instructions
1. Clone the repository: ```git clone https://github.com/slickFix/ECE143_Hotel_Recommendation_System.git```
2. Create new anaconda environment: ```conda create -n ece143 python=3.9```
3. Change directory by using ```cd ECE143_Hotel_Recommendation_System```
4. Execute following command in terminal: ```pip install -r requirements.txt```
5. Change directory into the src folder: ```cd src```
5. For running the complete pipeline execute: ```python main.py```
6. For using the app based recommendation execute: ```python app.py```

# Folder Structure
```
├── data
│   ├── bookings.csv
│   ├── combine_df.csv
│   ├── hotel_bookings.csv
│   ├── hotels.csv
│   └── user_bookings.csv
├── eda_jupyter_notebooks
│   ├── data_analysis.ipynb
│   └── data_cleaning.ipynb
├── Presentation.pdf
├── README.md
├── requirements.txt
└── src
    ├── app.py
    ├── data_cleaning.py
    ├── fastFM_model.py
    ├── hotel_trending.py
    ├── __init__.py
    ├── main.py
    ├── monthly_reccomendation.py
    ├── svdpp.py
    ├── util.py
    └── weighted_prediction.py
```

# Contributors*

* Zura Nebieridze
* Siddharth Shukla
* Xiaofeng Zhao
* Qihuang Chen
* Aman Gupta

*order according to course website
