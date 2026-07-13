# The Challenge

Cherry blossoms are often treated as a symbol of spring's arrival, but predicting exactly when they will bloom is a surprisingly difficult forecasting problem. The National Park Service has been quoted as saying that it is generally impossible to predict more than a week ahead of time. Peak bloom depends on a delicate combination of winter chilling, spring warming, local climate, geography, species differences, and year-to-year weather variability. This makes cherry blossom prediction a natural setting for testing statistical and machine learning methods under uncertainty.

The [international cherry blossom prediction competition](https://competition.statistics.gmu.edu/) is an annual competition that seeks to forecast the date of peak cherry blossom bloom across several locations across the world. Some of these locations, such as Kyoto or Washington D.C., have abundant historical data, while others, such as Vancouver or New York City, have fewer than a handful of data points. My model was selected as the winner for the 2026 competition, where I utilized a strategy leveraging both classical statistics and modern machine learning. 

# My Approach

In a loose sense, both classical statistics and modern machine learning handle scarce data settings by "borrowing strength." The thought here is to leverage partially shared information between data points to simultaneously capture their idiosyncrasies, while not overfitting the data. In statistics, this often takes the form of mixed-effect models in the Frequentist setting and hierarchical priors in the Bayesian setting. In machine learning, we can also see a type of "borrowing strength" in transfer learning. In this framework, machine learning models are first trained on a task where there exists abundant data and then fine-tuned or adapted to perform a secondary task with much less data.

The model therefore had two stages. First, I removed broad site-level and warming-related trends from the historical bloom dates. Second, I trained a neural network on much richer weather data and used the representations learned by that network to predict the remaining site-year deviations in bloom time.

In my model, I utilized both of the above-mentioned approaches. Each site we are asked to predict will generally have a different average peak time, and all sites are affected by warming trends. To handle this, I detrended the historical bloom data using a random-effects model with random intercepts per bloom site and a global slope term to capture warming. This means I am assuming uniform effects of long-term warming across relevant species and geographical sites for simplicity. Ultimately, this detrending approach allowed me to simultaneously account for warming trends and center predictions for each site.

In retrospect, I could have modified this approach to capture correlation among sites due to geographic proximity, perhaps making use of a Bayesian approach with a Gaussian process prior on site intercepts. The training data we were given was abundant in data from locations in South Korea, so this likely would have yielded some improvement in the model. Additionally, there may be room for improvement in accounting for between-species variation in the effects of warming.

To actually forecast the peak blossom dates beyond the first-order warming trends, I decided that historical weather and climate-index data would be the most useful. Cherry blossom blooming is largely driven by the local climate. Cherry blossoms need some number of chill hours during the winter, followed by warming in the spring, which is called forcing. Precipitation also plays a role, but ultimately these are the major factors determining bloom date. So the thought was that if I could train a model to learn useful temporal representations of local climate dynamics, those representations could help explain deviations from the long-term bloom trend.

In particular, I decided to use an approach rooted in modern representation learning. I first trained a stacked LSTM to predict next-day temperatures for each site in the historical cherry blossom dataset, using NASA POWER weather data, NOAA's ENSO anomaly datasets, and the coordinates of each site. At each index in the LSTM, today's high and low temperature, the ENSO anomaly value, and the coordinates of the site were fed into the neural network. The network was then trained to predict tomorrow's high and low temperature. The internal representations produced by this trained model encode information relevant to predicting temperature and allow me to leverage weather data at each site that is potentially much more abundant than the cherry blossom peak bloom data, which are extremely scarce for some sites.

I combined aggregated chill-hour proxy statistics and pooled LSTM representations from the winter (with exponential decay so the most recent dates matter most) and fed them as input to a small MLP to predict the residuals resulting from the detrended cherry blossom data.

By summing the mixed-effects projections and the predicted neural network residuals, I reconstructed the 2026 bloom dates. The resulting model achieved a Mean Absolute Error of approximately 4 days on the validation set and 5 days in the final evaluation. Although this error is larger than the strongest results from some prior years, the model was the winning entry for 2026, suggesting that this year's evaluated sites and weather conditions posed a particularly difficult forecasting problem. Given that predictions were submitted in mid February for blooms in late March to mid April, this represented competitive performance despite substantial data scarcity.

# Code
Because the competition asked for code to be submitted in a Quarto markdown file, I replicate some of the code here with brief explainations of their purpose:

This initial script loads reformats the relevant data, detrends the cherry blossom peak bloom time series and produces aggregated chill statistics.
```R
library(dplyr)
library(lubridate)
library(tidyr)
library(readr)
library(nasapower)
library(purrr)
library(lme4)

japan_df <- read.csv("./data/japan.csv")
japan_df <- japan_df[japan_df$location != "Japan/Kyoto",]

meteoswiss_df <- read.csv("./data/meteoswiss.csv")
south_korea_df <- read.csv("./data/south_korea.csv")
kyoto_df <- read.csv("./data/kyoto.csv")
liestal_df <- read.csv("./data/liestal.csv")
nyc_df <- read.csv("./data/nyc.csv")
vancouver_df <- read.csv("./data/vancouver.csv")
washingtondc_df <- read.csv("./data/washingtondc.csv")

pooled_df <- bind_rows(
  japan_df,
  kyoto_df,
  liestal_df,
  meteoswiss_df,
  nyc_df,
  south_korea_df,
  vancouver_df,
  washingtondc_df
)

write.csv(pooled_df, "./data/pooled_blooms.csv")


unique_sites <- pooled_df %>%
  distinct(location, .keep_all = TRUE) %>%
  select(location, lat, long)

write.csv(unique_sites, "./data/used_sites.csv")

# ENSO index
enso_weekly <- read_table(
  "https://psl.noaa.gov/data/correlation/nina34.anom.data",
  col_names = FALSE,
  skip = 1
)

enso_weekly[2:13] <- lapply(enso_weekly[2:13], as.double)
enso_weekly <- enso_weekly[-c(1,2,80,81,82),]

enso_weekly <- enso_weekly |>
  rename(year = X1) |>
  pivot_longer(-year, names_to="week", values_to="nino34_anom") |>
  mutate(
    week = as.integer(gsub("X", "", week)),
    date = as.Date(paste(year, week, 1, sep="-"), "%Y-%U-%u")
  ) |>
  select(date, nino34_anom) |>
  arrange(date)

enso_weekly <- enso_weekly[!(enso_weekly$nino34_anom < -90.0),]

enso_daily <- enso_weekly |>
  complete(date = seq(min(date), Sys.Date(), by="day")) |>
  fill(nino34_anom)

# Pull NASA power data
get_power_site <- function(lon, lat) {
  
  dat <- nasapower::get_power(
    community = "ag",
    pars = c("T2M_MAX", "T2M_MIN"),
    temporal_api = "daily",
    lonlat = c(lon, lat),
    dates = c("19810101", format(Sys.Date(), "%Y%m%d"))
  )
  
  dat %>%
    transmute(
      date = as.Date(YYYYMMDD),
      doy = DOY,
      tmax = T2M_MAX,
      tmin = T2M_MIN
    )
}

weather <- unique_sites %>%
  mutate(weather = map2(long, lat, get_power_site)) %>%
  unnest(weather)

# Merge climate and NASA power
processed_data <- weather %>%
  left_join(enso_daily, by="date") %>%
  left_join(unique_sites, by="location") %>%
  mutate(
    obs_year = year(date),
    obs_month = month(date),
    batch_year = if_else(obs_month >= 4, obs_year, obs_year - 1),
    batch_id = paste(location, batch_year, sep="_")
  ) %>%
  select(-obs_year, -obs_month, -lat.x, -long.x) %>%
  rename(long = long.y, lat = lat.y) %>%
  drop_na()

processed_data$date_numeric <-
  as.numeric(processed_data$date - min(processed_data$date))

write.csv(processed_data, "./data/processed_data.csv")

inference_data <- processed_data %>%
  filter(!(doy > 51 & doy < 91)) %>%
  mutate(bloom_year = batch_year + 1)

write.csv(inference_data, "./data/inference_training_data.csv")

filtered_bloom <- pooled_df %>% semi_join(inference_data, by = c("year" = "bloom_year", "location" = "location"))

bloom_regression <- filtered_bloom
bloom_regression$location <- as.factor(bloom_regression$location)

trend_model <- lmer(bloom_doy ~ year + (1 | location),
                    data = filtered_bloom)

summary(trend_model)

projections <- c("kyoto", "liestal", "newyorkcity", "vancouver", "washingtondc")
projection_2026 <- data.frame(
  year = 2026,
  location = factor(
    projections,
    levels = levels(bloom_regression$location)
  )
)
projection_2026$trend_doy <- predict(
  trend_model,
  newdata = projection_2026,
  re.form = NULL
)

write.csv(projection_2026, "./data/linear_projections_2026.csv")

filtered_bloom$trend_doy <- predict(trend_model, bloom_regression)

filtered_bloom$trend_residual <- (bloom_regression$bloom_doy - filtered_bloom$trend_doy)

write.csv(filtered_bloom, "./data/pooled_filtered_blooms.csv")

chill_range <- inference_data %>% 
  filter(doy <= 51 | doy >= 288) %>% 
  mutate(winter_indicator = (doy >= 288 | doy <= 51)) %>%
  mutate(
    winter_chill_min = winter_indicator & (tmin <= 7 & tmin >= 0),
    winter_chill_max = winter_indicator & (tmax <= 7 & tmax >= 0)
  ) %>%
  mutate(winter_chill_maxmin = winter_chill_max & winter_chill_min)
chill_stats <- chill_range %>% group_by(batch_id) %>%  
  summarise(
    chill_min = sum(winter_chill_min, na.rm = TRUE) / 100,
    chill_max = sum(winter_chill_max, na.rm = TRUE) / 100,
    chill_maxmin = sum(winter_chill_maxmin, na.rm = TRUE) / 100
  )

write.csv(chill_stats, "./data/chill_stats.csv")
```

I utilized this PyTorch model class I have written for previous projects to quickly prototype models.
```Python
import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(
        self,
        train_loader,
        loss_fn,
        optimizer=None,
        data_processor=None,
        logger=None,
        val_loader=None,
        checkpoint_fn=None,
        clean_up_fn=None,
        epochs=2000,
        lr=1e-3,
        tolerance=1e-4,
        patience=10,
        print_freq=1,
        checkpoint_freq=50,
        device="cpu",
    ):
        self.to(device)
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=lr)
        data_processor = data_processor or (lambda b: b)
        
        train_losses = []
        val_losses = []
        
        best_loss = float("inf")
        counter = 0

        for epoch in range(epochs):

            self.train()
            total_loss = 0

            for batch in train_loader:
                batch, target = data_processor(batch)

                optimizer.zero_grad()
                preds = self(batch.to(device))
                loss = loss_fn(preds, target.to(device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            train_losses.append(total_loss)

            val_loss = None
            if val_loader:
                self.eval()
                val_loss = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch, target = data_processor(batch)
                        preds = self(batch.to(device))
                        val_loss += loss_fn(preds, target.to(device)).item()

                val_loss /= len(val_loader)
                
                val_losses.append(val_loss)

            monitor = val_loss if val_loss is not None else total_loss

            if monitor < best_loss - tolerance:
                best_loss = monitor
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % print_freq == 0:
                msg = f"Epoch {epoch} | Train {total_loss/len(train_loader):.4f}"
                if val_loss:
                    msg += f" | Val {val_loss:.4f}"
                print(msg)

                if logger:
                    logger(self, epoch, total_loss, val_loss)
            
            if epoch % checkpoint_freq == 0:
                if checkpoint_fn:
                    checkpoint_fn(self, epoch, total_loss, val_loss)
            
        if clean_up_fn:
            clean_up_fn(self, train_losses, val_losses)
```

And then trained a simple stacked LSTM on the local climate data.
```Python
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ClimateDataset(Dataset):
    def __init__(
        self,
        df
    ):
        self.feature_cols = ["tmin", "tmax", "nino34_anom", "lat", "long"]
        self.samples = []
        
        dropped = 0
        
        for _, g in df.groupby("batch_id"):
            g = g.sort_values("date")
            
            # Drop leap days?
            feb29 = (g["date"].dt.month == 2) & (g["date"].dt.day == 29)
            g = g.loc[~feb29]
            
            if len(g) != 365: # If we don't have the right length we need to toss it out
                dropped += 1
                continue

            X = g[self.feature_cols].to_numpy(dtype=np.float32)

            inputs = X[:-1]
            targets = X[1:, :2]

            self.samples.append((inputs, targets))
        
        print(f"Dropped {dropped} samples due to incomplete sequence length")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def processor(batch):
    x, y = batch
    return x, y

def split_by_batch(df, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)

    batch_ids = df["batch_id"].unique()
    rng.shuffle(batch_ids)

    n_val = int(len(batch_ids) * val_frac)
    val_ids = set(batch_ids[:n_val])

    train_df = df[~df["batch_id"].isin(val_ids)].copy()
    val_df   = df[df["batch_id"].isin(val_ids)].copy()

    return train_df, val_df

class ClimateLSTM(Model):
    def __init__(self, lstm_depth=2, internal_dim=128):
        super().__init__()
        
        self.mu = nn.Parameter(torch.zeros(5))
        self.log_sigma = nn.Parameter(torch.zeros(5))
        
        self.enc = nn.Linear(5, 32)
        self.proj = nn.Linear(32, 64)
        self.lstm = nn.LSTM(64, internal_dim, num_layers=lstm_depth, batch_first=True)
        self.dec = nn.Linear(internal_dim, 2)
        
    def forward(self, data):
        normed = (data - self.mu) / torch.exp(self.log_sigma)
        encoding = self.enc(normed).relu()
        proj = self.proj(encoding)
        
        out, _ = self.lstm(proj)
        
        return out, self.dec(out)
        

def loss_fn(model_out, target):
    loss_fn = torch.nn.MSELoss()
    
    h, out = model_out
    
    return loss_fn(out, target)

def cleanup(m, train_losses, val_losses):
    torch.save(m, "climate_model.pt")


climate_df = pd.read_csv("./data/processed_data.csv")
climate_df["date"] = pd.to_datetime(climate_df["date"])

train_df, val_df = split_by_batch(
    climate_df,
    val_frac=0.2,
    seed=42
)

train_dataset = ClimateDataset(
    train_df
)

val_dataset = ClimateDataset(
    val_df
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

climate_model = ClimateLSTM()

climate_model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    data_processor=processor,
    clean_up_fn=cleanup
)
```

I then construct the final training data for the prediction MLP.
```Python
import torch
import numpy as np
import pandas as pd

climate_model = torch.load("climate_model.pt", weights_only=False)

df = pd.read_csv("./data/inference_training_data.csv", index_col=0)

feature_cols = ["tmin", "tmax", "nino34_anom", "lat", "long"]

samples = []
batch_meta = []

for batch_id, g in df.groupby("batch_id"):
    g = g.sort_values("date")

    if len(g) < 100:
        continue

    X = g[feature_cols].to_numpy(dtype=np.float32)
    inputs = torch.tensor(X)

    samples.append(inputs)

    batch_meta.append({
        "batch_id": batch_id,
        "location": g["location"].iloc[0],
        "bloom_year": g["bloom_year"].iloc[0]
    })

latent_rows = []

with torch.no_grad():
    for s, meta in zip(samples, batch_meta):
        hidden_repr, _ = climate_model(s)

        last_40 = hidden_repr[-40:, :].cpu().numpy()

        flat = last_40.reshape(-1)

        row = {
            "batch_id": meta["batch_id"],
            "location": meta["location"],
            "bloom_year": meta["bloom_year"]
        }

        for i, val in enumerate(flat):
            row[f"latent_{i}"] = val

        latent_rows.append(row)

latent_df = pd.DataFrame(latent_rows)

chill_df = pd.read_csv("./data/chill_stats.csv", index_col=0)

latent_df = latent_df.merge(chill_df, on="batch_id", how="left")

bloom_df = pd.read_csv("./data/pooled_filtered_blooms.csv", index_col=0)

latent_df = latent_df.merge(
bloom_df[["location", "year", "trend_doy", "trend_residual", "bloom_doy"]],
left_on=["location", "bloom_year"],
right_on=["location", "year"],
how="left"
)

latent_df = latent_df.drop(columns=["year"])

latent_df.to_csv("./data/prediction_training_data.csv", index=False)

print(latent_df.head())
```

And then train the prediction head.
```Python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BlossomDataset(Dataset):
    def __init__(
        self,
        df
    ):
        self.samples = []

        for index, row in df.iterrows():
            X = row[3:-3].to_numpy(dtype=np.float32)
            Y = row.iloc[-2]

            self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class BlossomPredictor(Model):
    def __init__(self):
        super().__init__()

        self.raw_lambda = nn.Parameter(torch.tensor(-1.0))

        self.norm = nn.LayerNorm(131)
        
        self.main = nn.Sequential(
            nn.Linear(131, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.shortcut_proj = nn.Linear(131, 64)

        self.out = nn.Linear(64, 1)

    def forward(self, data):

        neg_lambda = -nn.functional.softplus(self.raw_lambda)

        # Perform pooling with exponential decay
        steps = torch.arange(40, device=data.device).float()
        weights = torch.exp(neg_lambda * steps) 
        
        weights = weights / weights.sum()
        
        exp_pool = torch.zeros([data.shape[0], 128], device=data.device)
        for i in range(40):
            exp_pool += weights[i] * data[:, i*128:(i+1)*128]

        x = torch.cat([exp_pool, data[:, -3:]], dim=-1)

        x = self.norm(x)
        x = self.main(x) + self.shortcut_proj(x)

        return self.out(x)

def split_dataset(df, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)

    nrows = len(df)
    row_ids = np.arange(nrows)
    rng.shuffle(row_ids)

    n_val = int(nrows * val_frac)

    val_ids = row_ids[:n_val]
    train_ids = row_ids[n_val:]

    train_df = df.iloc[train_ids].copy()
    val_df = df.iloc[val_ids].copy()

    return train_df, val_df

def processor(batch):
    x, y = batch
    return x, y

def loss_fn(model_out, target):
    loss_fn = nn.MSELoss()

    output = model_out

    return loss_fn(output.squeeze(-1), target)


def cleanup(m, train_losses, val_losses):
    torch.save(m, "blossom_model.pt")

df = pd.read_csv("./data/prediction_training_data.csv")
filtered_df = df.dropna()
print(f"Dropped {len(df) - len(filtered_df)} entries due to NA values")

train_df, val_df = split_dataset(filtered_df)

train_dataset = BlossomDataset(train_df)
val_dataset = BlossomDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=250,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=250,
    shuffle=False
)

blossom_model = BlossomPredictor()

blossom_model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    data_processor=processor,
    clean_up_fn=cleanup,
    patience=25
)

# Now we will assess the performance based on mean absolute error.
eval_df = val_df.reset_index()
error = 0.0
with torch.no_grad():
    for index, row in eval_df.iterrows():
        prediction = np.round(row.loc["trend_doy"] + blossom_model(val_dataset[index][0].unsqueeze(0)).numpy())
        error += abs(row.loc["bloom_doy"] - prediction)
print(f"Mean absolute error: {error / len(val_dataset)}")
```

We finally produce our predictions for 2026
```Python 
prediction_cities = df[df["batch_id"].isin(["newyorkcity_2025", "liestal_2025", "vancouver_2025", "washingtondc_2025", "kyoto_2025"])] # Batch IDs contain the year when the time series starts

projections = pd.read_csv("./data/linear_projections_2026.csv", index_col = 0)

residual_predictions = None

with torch.no_grad():
    prediction_tensor = torch.tensor(prediction_cities.iloc[:, 3:-3].to_numpy(), dtype=torch.float32)
    residual_predictions = blossom_model(prediction_tensor).numpy().squeeze()

print(f"Residual predictions {residual_predictions}")

for i in range(5):
    loc = prediction_cities["location"].iloc[i]
    lp = projections.loc[
        projections["location"] == loc,
        "trend_doy"
    ].iloc[0]
    print(f"{loc:<25} {np.round(lp + residual_predictions[i]):>12.4f}")
```