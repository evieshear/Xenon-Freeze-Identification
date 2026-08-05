import pandas as pd
import glob
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

# cutoff_214_start = datetime.datetime(2026, 2, 13, 12, 0, 0, 0)
# cutoff_214_end = datetime.datetime(2026, 2, 15, 11, 59, 59, 59)
# cutoff_214_good = datetime.datetime(2026, 2, 14, 0, 0, 0)
# cutoff_214_bad = 

cutoff_309_1 = datetime.datetime(2026, 3, 7, 12, 0, 0)
cutoff_309_2 = datetime.datetime(2026, 3, 8, 2, 0, 0)

cutoff_309_3 = datetime.datetime(2026, 3, 8, 6, 10, 0)
cutoff_309_4 = datetime.datetime(2026, 3, 10, 5, 59, 59)


cutoff_318_1 = datetime.datetime(2026, 3, 17, 17, 17, 0)
cutoff_318_2 = datetime.datetime(2026, 3, 18, 4, 30, 0)

cutoff_318_3 = datetime.datetime(2026, 3, 18, 12, 0, 0)
cutoff_318_4 = datetime.datetime(2026, 3, 18, 13, 0, 0)


cutoff_506_1 = datetime.datetime(2026, 5, 6, 10, 55, 0)
cutoff_506_2 = datetime.datetime(2026, 5, 6, 16, 40, 0)

cutoff_506_3 = datetime.datetime(2026, 5, 6, 19, 20, 0)
cutoff_506_4 = datetime.datetime(2026, 5, 6, 21, 59, 59)


cutoff_322_1 = datetime.datetime(2026, 3, 19, 10, 52, 0)
cutoff_322_2 = datetime.datetime(2026, 3, 27, 14, 7, 0)


cutoff_214_3 = datetime.datetime(2026, 2, 14, 16, 40, 0)
cutoff_214_4 = datetime.datetime(2026, 2, 15, 12, 0, 0)

csv_dir = r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\RQs\\"

def extract_csv_data(filename, start_datetime, end_datetime, quality=-1, freezeID=-1):

    # parse and slice dataframe between given times

    temp_df = pd.read_csv(csv_dir + filename + ".csv", index_col=0, parse_dates=['time'])
    df_cut = temp_df[(start_datetime <= temp_df['time']) & (temp_df['time'] <= end_datetime)].copy()

    if quality != -1:
        df_cut['quality']=quality
        df_cut['freezeID']=freezeID

    return(df_cut)

def group_foci(df, quality = -1, freezeID = -1):
    # sort dataframe by time

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)


    # group images by focus if they are close together in time

    required_focuses = {0, 2, 4, 6, 8, 10}
    max_span = pd.Timedelta(seconds=120)   # choose an appropriate value

    groups = []
    current = []
    seen_focuses = set()

    for _, row in df.iterrows():

        if not current:
            current = [row]
            seen_focuses = {row["focus"]}
            continue

        first_time = current[0]["time"]

        # Too much time has elapsed → abandon this partial group
        if row["time"] - first_time > max_span:
            current = [row]
            seen_focuses = {row["focus"]}
            continue

        # Duplicate focus → restart from this image
        if row["focus"] in seen_focuses:
            current = [row]
            seen_focuses = {row["focus"]}
            continue

        current.append(row)
        seen_focuses.add(row["focus"])

        if seen_focuses == required_focuses:
            groups.append(pd.DataFrame(current))
            current = []
            seen_focuses = set()

    # Turn dataframes in "groups" into a single row of a dataframe

    records = []
    for sub in groups:

        record = {}
        for _, row in sub.iterrows():
            f = row["focus"]
            if f == 0 :
                record["time"] = row["time"]

            record[f"pixel_std_{f}"] = row["pixel_std"]
            record[f"entropy_{f}"] = row["entropy"]
            record[f"laplace_var_{f}"] = row["laplace_var"]
            record[f"brightness_{f}"] = row["brightness"]
            record[f"ssim_{f}"] = row["ssim"]
        
        if quality != -1:
            record["quality"] = quality
        if freezeID != -1:
            record["freezeID"] = freezeID
        records.append(record)
    df_output = pd.DataFrame(records)

    return(df_output)

def helper(filename, start_datetime, end_datetime, quality, freezeID):
    return(group_foci(
        extract_csv_data(filename, start_datetime, end_datetime),
        quality, freezeID
        )
    )

if __name__ == "__main__":
    df = pd.concat([
        helper('0309', cutoff_309_1, cutoff_309_2, 0, 0),
        helper('0309', cutoff_309_3, cutoff_309_4, 1, 0),
        helper('0318', cutoff_318_1, cutoff_318_2, 0, 1),
        helper('0318', cutoff_318_3, cutoff_318_4, 1, 1),
        helper('0506', cutoff_506_1, cutoff_506_2, 0, 2),
        helper('0506', cutoff_506_3, cutoff_506_4, 1, 2),
        helper('0322', cutoff_322_1, cutoff_322_2, 0, 3),
        helper('0214', cutoff_214_3, cutoff_214_4, 1, 4)
        ])
    df = df.reset_index()

    # df = pd.concat([
    #         extract_csv_data('0309', cutoff_309_1, cutoff_309_2, 0, 0),
    #         extract_csv_data('0309', cutoff_309_3, cutoff_309_4, 1, 0),
    #         extract_csv_data('0318', cutoff_318_1, cutoff_318_2, 0, 1),
    #         extract_csv_data('0318', cutoff_318_3, cutoff_318_4, 1, 1),
    #         extract_csv_data('0506', cutoff_506_1, cutoff_506_2, 0, 2),
    #         extract_csv_data('0506', cutoff_506_3, cutoff_506_4, 1, 2),
    #         extract_csv_data('0322', cutoff_322_1, cutoff_322_2, 0, 3),
    #         extract_csv_data('0214', cutoff_214_3, cutoff_214_4, 1, 4)
    #         ])
    # df = df.reset_index()


    df.to_csv("RQs.csv", index=True)