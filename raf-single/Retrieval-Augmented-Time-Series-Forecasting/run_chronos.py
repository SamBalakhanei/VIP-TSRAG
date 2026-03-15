# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is based on https://github.com/amazon-science/chronos-forecasting/
# Original Copyright (c) 2023 Abdul Fatir
# Modified by Kutay Tire on October 2024


import logging
from pathlib import Path
from typing import Iterable, Optional


import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from time_series_utils import augment_time_series, denormalize_predictions, normalize_context
from chronos_local import ChronosPipeline


tickers = [
"AOMR","REGCP","WRBY","FVRR","NET","AVO","PROF","AQB","PMTS","ESQ","LFT","DUOL","EDBL","HLLY","DINO","VRTL","LIVN","KMPB","HL","LXP",
"STG","OPK","BR","PHAT","SERA","RWAYL","NLY","TELO","CAKE","ABUS","DKNG","DLX","RXST","PRGS","BDTX","CBOE","WLDN","YOU","LANDM","GME",
"EGO","OCCIN","SVRN","FAT","MSFT","UYSCU","SABS","GALT","CDIO","ROIV","DEFT","USB","ELS","SDOT","RDHL","FLYE","BLIV","NWL","ECCW","CNDT",
"ECCV","RKT","CAN","TRUP","ABXL","JBLU","FLGT","USIO","HAL","PCSC","CRK","ALNY","FUFU","SHMD","ULY","AGYS","JBSS","CTVA","NNN","LPRO",
"RBNE","BIPJ","CSTM","ARW","LPBBU","BDL","SWKS","LZMH","M","JFIN","CTRI","JOYY","TGNA","PCVX","ULTA","WINA","TMP","OOMA","ALUR","ATEN",
"MEGI","OAKU","OTEX","EZRA","GETY","TCBC","HGV","HVIIU","DUETU","GNS","PCLA","YEXT","MHUA","FSK","NRDS","NIC","BBD","ALB","BL","ETHZ",
"CALY","UK","AXS","FARM","NVS","TRNS","MRKR","ELA","BOF","IMNN","TCBIO","BXP","GLBZ","TFSL","CYCN","ITP","BRW","USCB","ING","HTT",
"PMTV","RYAM","EEX","DXYZ","CATO","PBI","XPL","GLBS","ADP","PNFPP","SOUN","IIIV","AX","PSIG","ANF","SMID","CTRM","CWH","AFGC","CNC",
"TAP","CRBG","FICO","SRAX","MRNO","TBN","PLRZ","INLX","CVX","DBVT","UI","MGNI","NEXM","VHAI","COHN","VZ","FISI","DCOM","IART","TFPM",
"LCNB","MVBF","VIVK","ENPH","NRIX","BV","BANF","UFCS","EXLS","SSBI","UCB","ASTE","OXLCI","SLNG","RR","SKWD","AMPX","ITIC","PFBC","RL",
"JZXN","AMBR","DLNG","WTW","BOW","IPDN","KRYS","E","FTEK","WIMI","KLTO","VFS","SAY","PALI","AUPH","BMEZ","APEI","SPIR","EGAN","EPWK",
"CLB","UMBF","CIX","CBRE","ATEX","BEAT","BETR","INDI","NUE","HOUS","IMOS","AMZN","WPC","IMNM","ARCO","TCBK","HGLB","DVN","ARMN","CRUS",
"BCAX","BPYPM","XGN","PHR","CBAT","TOL","EQIX","VRA","SSB","XBIO","BNAI","ADCT","HBANL","LB","XIFR","BELFB","HNI","CCRN","CHCO","NE",
"MRUS","CHR","CAPR","SDHC","PRAA","HYLN","RWAYZ","AN","ELME","KRG","PRDO","INKT","TGB","CAE","EFOI","ALFUU","MXCT","ELUT","CMG","THRM",
"GPI","DFSC","OS","CNX","UFG","AA","CF","PGYWW","SONY","MVO","TPG","SKYH","CMSA","TRV","GIFT","RNA","LFMDP","CODX","ARQT","AME","XTIA",
"CPHC","SPFI","ERIE","LNSR","BOSC","PNFP","SOHON","AIFF","RLYB","APAM","IDT","PENG","CVV","LSAK","LIN","CPS","BKD","VYLD","TPGXL","SID",
"TOVX","DDS","HCKT","LBTYK","VLTO","APLT","CGON","DCOMP","USAC","SNDA","KXIN","TRDA","DGLY","MGIH","TACT","SEAT","LSPD","GEF-B","FUND",
"ALEX","ASPN","RAVE","WT","FHB","ESBA","IOTR","HIW","SIBN","MWA","MCRB","OPAD","SIGIP","BKNG","FTAIM","AVX","MITN","DGNX","AQN","BSAC",
"SAGT","GAINZ","TNDM","ONBPO","MCY","ATII","ARL","APVO","GDHG","API","REVB","ISPC","UUU","MBC","AFCG","MP","R","AMOD","VG","HFWA","PRA",
"DK","AGI","DYCQU","CRS","DHCNI","COLB","BVFL","ENLT","JWEL","ANL","PRE","PEW","IRDM","PNRG","AIRI","HG","GWRE","UGRO","INBKZ","DKS",
"DFH","SNFCA","PHGE","GOODO","SNOW","FUTU","RMBS","CRT","BIP","EVEX","TGTX","WILC","CMCM","RHI","RNXT","XYL","SPCB","PDSB","OPAL","RGA",
"NFBK","CNO","KULR","CNA","HTFC","BGS","SOPA","DDT","SIMAU","OMER","CHMG","FCPT","CBIO","LOKVU","INEO","QETAU","LOAR","BDX","NWFL","CMMB",
"ROCK","OMEX","ICHR","FFIV","QFIN","NHPBP","BF-A","NL","DOX","XPER","NTB","NEGG","MPAA","SMMT","EMPD","VTSI","SAIL","SGD","CNR","GRDX",
"NCMI","HY","SPMC","OVV","VRME","FSTR","MGNX","PUK","UA","BIDU","RSSS","NBR","NFGC","NEXA","EVR","ZBRA","TFX","ARI","FIZZ","FBP","PLRX",
"IXHL","SLVM","NMG","ESE","ETH","TNMG","UNFI","LQDA","RCKY","CVCO","AKA","TMUS","QSR","SSII","JHG","WSBF","GVH","PFIS"
]


app = typer.Typer(pretty_exceptions_enable=False)




# Taken from pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
   "WEEKDAY": "D",
   "EOM": "M",
   "BME": "M",
   "SME": "M",
   "BQS": "Q",
   "QS": "Q",
   "BQE": "Q",
   "BQE-DEC": "Q",
   "BQE-JAN": "Q",
   "BQE-FEB": "Q",
   "BQE-MAR": "Q",
   "BQE-APR": "Q",
   "BQE-MAY": "Q",
   "BQE-JUN": "Q",
   "BQE-JUL": "Q",
   "BQE-AUG": "Q",
   "BQE-SEP": "Q",
   "BQE-OCT": "Q",
   "BQE-NOV": "Q",
   "MS": "M",
   "D": "D",
   "B": "B",
   "min": "min",
   "s": "s",
   "ms": "ms",
   "us": "us",
   "ns": "ns",
   "h": "h",
   "QE": "Q",
   "QE-DEC": "Q-DEC",
   "QE-JAN": "Q-JAN",
   "QE-FEB": "Q-FEB",
   "QE-MAR": "Q-MAR",
   "QE-APR": "Q-APR",
   "QE-MAY": "Q-MAY",
   "QE-JUN": "Q-JUN",
   "QE-JUL": "Q-JUL",
   "QE-AUG": "Q-AUG",
   "QE-SEP": "Q-SEP",
   "QE-OCT": "Q-OCT",
   "QE-NOV": "Q-NOV",
   "YE": "Y",
   "YE-DEC": "Y-DEC",
   "YE-JAN": "Y-JAN",
   "YE-FEB": "Y-FEB",
   "YE-MAR": "Y-MAR",
   "YE-APR": "Y-APR",
   "YE-MAY": "Y-MAY",
   "YE-JUN": "Y-JUN",
   "YE-JUL": "Y-JUL",
   "YE-AUG": "Y-AUG",
   "YE-SEP": "Y-SEP",
   "YE-OCT": "Y-OCT",
   "YE-NOV": "Y-NOV",
   "W": "W",
   "ME": "M",
   "Y": "Y",
   "BYE": "Y",
   "BYE-DEC": "Y",
   "BYE-JAN": "Y",
   "BYE-FEB": "Y",
   "BYE-MAR": "Y",
   "BYE-APR": "Y",
   "BYE-MAY": "Y",
   "BYE-JUN": "Y",
   "BYE-JUL": "Y",
   "BYE-AUG": "Y",
   "BYE-SEP": "Y",
   "BYE-OCT": "Y",
   "BYE-NOV": "Y",
   "YS": "Y",
   "BYS": "Y",
   "QS-JAN": "Q",
   "QS-FEB": "Q",
   "QS-MAR": "Q",
   "QS-APR": "Q",
   "QS-MAY": "Q",
   "QS-JUN": "Q",
   "QS-JUL": "Q",
   "QS-AUG": "Q",
   "QS-SEP": "Q",
   "QS-OCT": "Q",
   "QS-NOV": "Q",
   "QS-DEC": "Q",
   "BQS-JAN": "Q",
   "BQS-FEB": "Q",
   "BQS-MAR": "Q",
   "BQS-APR": "Q",
   "BQS-MAY": "Q",
   "BQS-JUN": "Q",
   "BQS-JUL": "Q",
   "BQS-AUG": "Q",
   "BQS-SEP": "Q",
   "BQS-OCT": "Q",
   "BQS-NOV": "Q",
   "BQS-DEC": "Q",
   "YS-JAN": "Y",
   "YS-FEB": "Y",
   "YS-MAR": "Y",
   "YS-APR": "Y",
   "YS-MAY": "Y",
   "YS-JUN": "Y",
   "YS-JUL": "Y",
   "YS-AUG": "Y",
   "YS-SEP": "Y",
   "YS-OCT": "Y",
   "YS-NOV": "Y",
   "YS-DEC": "Y",
   "BYS-JAN": "Y",
   "BYS-FEB": "Y",
   "BYS-MAR": "Y",
   "BYS-APR": "Y",
   "BYS-MAY": "Y",
   "BYS-JUN": "Y",
   "BYS-JUL": "Y",
   "BYS-AUG": "Y",
   "BYS-SEP": "Y",
   "BYS-OCT": "Y",
   "BYS-NOV": "Y",
   "BYS-DEC": "Y",
   "Y-JAN": "Y-JAN",
   "Y-FEB": "Y-FEB",
   "Y-MAR": "Y-MAR",
   "Y-APR": "Y-APR",
   "Y-MAY": "Y-MAY",
   "Y-JUN": "Y-JUN",
   "Y-JUL": "Y-JUL",
   "Y-AUG": "Y-AUG",
   "Y-SEP": "Y-SEP",
   "Y-OCT": "Y-OCT",
   "Y-NOV": "Y-NOV",
   "Y-DEC": "Y-DEC",
   "Q-JAN": "Q-JAN",
   "Q-FEB": "Q-FEB",
   "Q-MAR": "Q-MAR",
   "Q-APR": "Q-APR",
   "Q-MAY": "Q-MAY",
   "Q-JUN": "Q-JUN",
   "Q-JUL": "Q-JUL",
   "Q-AUG": "Q-AUG",
   "Q-SEP": "Q-SEP",
   "Q-OCT": "Q-OCT",
   "Q-NOV": "Q-NOV",
   "Q-DEC": "Q-DEC",
   "W-MON": "W-MON",
   "W-TUE": "W-TUE",
   "W-WED": "W-WED",
   "W-THU": "W-THU",
   "W-FRI": "W-FRI",
   "W-SAT": "W-SAT",
   "W-SUN": "W-SUN",
}




def to_gluonts_univariate(hf_dataset: datasets.Dataset):
   series_fields = [
       col
       for col in hf_dataset.features
       if isinstance(hf_dataset.features[col], datasets.Sequence)
   ]
   series_fields.remove("timestamp")
   dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
   dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
   dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)


   gts_dataset = []
   for hf_entry in hf_dataset:
       for field in series_fields:
           gts_dataset.append(
               {
                   "start": pd.Period(
                       hf_entry["timestamp"][0],
                       freq=dataset_freq,
                   ),
                   "target": hf_entry[field],
               }
           )
   assert len(gts_dataset) == dataset_length


   return gts_dataset


def get_test_sequences(test_input, test_len):
   test_instance_list = []
   for instance in test_input:
       test_instance = instance["target"][-test_len:]
       test_values = test_instance.astype(float) 
       test_values_tensor = torch.tensor(test_values)
       test_instance_list.append(test_values_tensor)


   return test_instance_list


def load_and_split_dataset(backtest_config: dict, ticker: str):


   hf_repo = backtest_config["hf_repo"]
   offset = backtest_config["offset"]
   prediction_length = backtest_config["prediction_length"]
   num_rolls = backtest_config["num_rolls"]
   max_history = backtest_config["max_history"]
   distance = backtest_config["distance"]


   ds = datasets.load_dataset(hf_repo, split="train", token="<token>")


   df = ds.to_pandas()


   df = df[df["symbol"] == ticker].copy()


   if df.shape[0] == 0:
       raise ValueError(f"No data for ticker {ticker}")


   df["date"] = pd.to_datetime(df["date"])
   df = df.sort_values("date")


   target = df["adj_close"].astype(float).values


   gts_dataset = [
       {
           "start": pd.Period(df["date"].iloc[0], freq="D"),
           "target": target,
       }
   ]


   train_df = gts_dataset
   test_df = gts_dataset


   _, test_template = split(test_df, offset=offset)


   test_data = test_template.generate_instances(
       prediction_length,
       windows=num_rolls,
       distance=distance,
       max_history=max_history,
   )


   return test_data, train_df




def generate_sample_forecasts(
   train_df,
   augment: bool,
   top_n: int,
   test_data_input: Iterable,
   pipeline: ChronosPipeline,
   prediction_length: int,
   batch_size: int,
   num_samples: int,
   **predict_kwargs,
):
   # Generate forecast samples
   forecast_samples = []
   for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
       context = [torch.tensor(entry["target"]) for entry in batch]


       if augment:
           context,mean_std_values = augment_time_series(train_df, pipeline, context, prediction_length, top_n)
       else:
           context,mean_std_values = normalize_context(context)
      
       prediction = pipeline.predict(
           context,
           prediction_length=prediction_length,
           num_samples=num_samples,
           limit_prediction_length=False,
           **predict_kwargs,
       )
       prediction = denormalize_predictions(prediction, mean_std_values)
       forecast_samples.append(prediction)
      
   # Convert forecast samples into gluonts SampleForecast objects
   sample_forecasts = []
   forecast_samples = np.concatenate(forecast_samples)


   for item, ts in zip(forecast_samples, test_data_input):
       forecast_start_date = ts["start"] + len(ts["target"])
       sample_forecasts.append(
           SampleForecast(samples=item, start_date=forecast_start_date)
       )


   return sample_forecasts


@app.command()
def main(
   config_file: str,
   result_file: str,
   augment: bool = False,
):


   with open(config_file) as f:
       configs = yaml.safe_load(f)


   pipeline = ChronosPipeline.from_pretrained(
       "amazon/chronos-t5-base",
       device_map="auto",
       torch_dtype=torch.bfloat16,
   )


   all_results = []


   for config in configs:


       prediction_length = config["prediction_length"]


       for ticker in tickers:


           print(f"\nRunning {ticker}")


           try:


               test_data, train_df = load_and_split_dataset(
                   backtest_config=config,
                   ticker=ticker
               )


               forecasts = generate_sample_forecasts(
                   train_df=train_df,
                   augment=augment,
                   top_n=5,
                   test_data_input=test_data.input,
                   pipeline=pipeline,
                   prediction_length=prediction_length,
                   batch_size=32,
                   num_samples=20,
               )


               metrics = (
                   evaluate_forecasts(
                       forecasts,
                       test_data=test_data,
                       metrics=[
                           MASE(),
                           MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                       ],
                       batch_size=5000,
                   )
                   .reset_index(drop=True)
                   .to_dict(orient="records")
               )


               metrics_df = pd.DataFrame(metrics).rename(
           {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
           axis="columns",
       )


               metrics_df["ticker"] = ticker
               metrics_df = metrics_df.iloc[:, [2, 0, 1]]


               all_results.append(metrics_df)


           except Exception as e:


               print(f"Skipping {ticker}: {e}")


   if len(all_results) == 0:
       print("No results generated")
       return


   results_df = pd.concat(all_results, ignore_index=True)


   results_df.to_csv(result_file, index=False)


   print("\nFinished experiments")
   print(f"Saved results to {result_file}")


if __name__ == "__main__":
   logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
   logger = logging.getLogger("Chronos Evaluation")
   logger.setLevel(logging.INFO)
   app()


