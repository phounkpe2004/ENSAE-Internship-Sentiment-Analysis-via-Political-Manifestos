import matplotlib.pyplot as plt
import pandas as pd


def plot_country_trajectories(theta_summary: pd.DataFrame, countries: list[str], conf_intervals: bool = False):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, ax = plt.subplots(figsize=(15, 10))

    for country in countries:
        subset = theta_summary[theta_summary["country"] == country].sort_values("year")
        ax.plot(subset["year"], subset["mean_theta"], label=country, marker='o')
        if conf_intervals:
            ax.fill_between(subset["year"], subset["q025"], subset["q975"], alpha=0.15)

    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Année")
    ax.set_ylabel(r"$\theta_{it}$")
    ax.set_title("Évolution des points idéaux")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    
    return fig



def plot_distance_to_USA(theta_summary: pd.DataFrame, countries: list[str], return_figure: bool = False):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, ax = plt.subplots(figsize=(15, 10))

    usa = (
        theta_summary[theta_summary["country"] == "USA"]
        .groupby("year")["mean_theta"]
        .mean()
    )

    results = []

    for country in countries:
        subset = theta_summary[theta_summary["country"] == country].sort_values("year").copy()
        subset["distance_to_usa"] = subset["mean_theta"] - subset["year"].map(usa)
        ax.plot(subset["year"], subset["distance_to_usa"], label=country, marker='o')
        results.append(subset)

    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Année")
    ax.set_ylabel(r"$\Delta \theta_{it}$")
    ax.set_title("Distance idéologique par rapport aux USA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    df = pd.concat(results, ignore_index=True)[["country", "year", "distance_to_usa"]] if results else pd.DataFrame()

    if return_figure:
        return fig, df
    return df

def plot_distance_sentiment(data, target_country = "AUS"):
    
    subset = data[data["country"] == target_country].sort_values("year")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Y-Axis: Distance
    color1 = 'tab:blue'
    ax1.set_xlabel('Année')
    ax1.set_ylabel('Distance idéologique aux USA', color=color1)
    line1 = ax1.plot(subset['year'], subset['distance_to_usa'], color=color1, marker='o', label='Distance')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Secondary Y-Axis: Country Sentiment towards USA
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Sentiment envers les USA', color=color2)
    line2 = ax2.plot(subset['year'], subset['country_sentiment_towards_usa'], color=color2, marker='s', linestyle='--', label='Sentiment')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f"Évolution de la distance et du sentiment envers les USA ({target_country})")
    plt.grid(True, alpha=0.3)
    plt.show()