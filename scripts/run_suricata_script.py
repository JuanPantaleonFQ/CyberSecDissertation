import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === SETUP ===

# Define output directories
output_dir = Path("merged_output")
output_dir.mkdir(parents=True, exist_ok=True)

base_dir = Path("output/suricata_logs")
all_alerts = []

# === READ AND EXTRACT ALERTS ===

# Loop through each eve.json file
for subfolder in base_dir.glob("output_pcap*/eve.json"):
    pcap_name = subfolder.parent.name
    with open(subfolder, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("event_type") == "alert":
                all_alerts.append({
                    "pcap": pcap_name,
                    "timestamp": entry.get("timestamp"),
                    "src_ip": entry.get("src_ip"),
                    "src_port": entry.get("src_port"),
                    "dest_ip": entry.get("dest_ip"),
                    "dest_port": entry.get("dest_port"),
                    "proto": entry.get("proto"),
                    "alert_signature": entry["alert"]["signature"],
                    "alert_category": entry["alert"]["category"],
                    "alert_severity": entry["alert"]["severity"]
                })

# Convert to DataFrame
merged_alerts_df = pd.DataFrame(all_alerts)

# Save to CSV
csv_output_path = output_dir / "merged_suricata_alerts.csv"
merged_alerts_df.to_csv(csv_output_path, index=False)

# Check if any alerts were found
if not merged_alerts_df.empty and "pcap" in merged_alerts_df.columns:
    # Plot: Alerts per PCAP
    plt.figure(figsize=(10, 5))
    sns.countplot(data=merged_alerts_df, y="pcap", order=merged_alerts_df["pcap"].value_counts().index, palette="Blues_d")
    plt.title("Number of Alerts per PCAP")
    plt.xlabel("Alert Count")
    plt.ylabel("PCAP File")
    plt.tight_layout()
    plt.savefig(output_dir / "alerts_per_pcap.png")
    plt.close()

    # Plot: Top 10 Alert Categories
    top_categories = merged_alerts_df["alert_category"].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
    plt.title("Top 10 Alert Categories")
    plt.xlabel("Alert Count")
    plt.ylabel("Alert Category")
    plt.tight_layout()
    plt.savefig(output_dir / "top_alert_categories.png")
    plt.close()

    # Plot: Top 10 Alert Signatures
    top_signatures = merged_alerts_df["alert_signature"].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_signatures.values, y=top_signatures.index, palette="magma")
    plt.title("Top 10 Alert Signatures")
    plt.xlabel("Alert Count")
    plt.ylabel("Alert Signature")
    plt.tight_layout()
    plt.savefig(output_dir / "top_alert_signatures.png")
    plt.close()

    # Summary
    print(f"✅ Merged {len(merged_alerts_df)} alerts from {merged_alerts_df['pcap'].nunique()} PCAP files.")
    print(f"📄 CSV saved to: {csv_output_path}")
    print(f"📊 Charts saved in folder: {output_dir}")
else:
    print("⚠️ No alerts found or missing 'pcap' column in alert data.")
    print(f"🛑 Check if your eve.json files contain alert entries.")
