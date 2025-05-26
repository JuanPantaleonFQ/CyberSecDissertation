import json
import pandas as pd
from pathlib import Path

# === SETUP ===

# Define where the merged CSV will be saved
output_dir = Path("merged_output")
output_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

# Define the directory where all your Suricata outputs are stored
base_dir = Path("output/suricata_logs")

# List to store all extracted alert dictionaries
all_alerts = []

# === READ AND EXTRACT ALERTS ===

# Loop through every eve.json in folders named output_pcap*
for subfolder in base_dir.glob("output_pcap*/eve.json"):
    pcap_name = subfolder.parent.name  # Example: output_pcap4

    # Open the JSON file line by line
    with open(subfolder, 'r') as f:
        for line in f:
            entry = json.loads(line)  # Parse each line as a JSON object

            # Only process if it's an alert
            if entry.get("event_type") == "alert":
                # Extract key alert fields into a dictionary
                all_alerts.append({
                    "pcap": pcap_name,  # Keep track of which PCAP triggered this alert
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

# === SAVE TO CSV ===

# Convert the list of alerts into a DataFrame (like a table)
merged_alerts_df = pd.DataFrame(all_alerts)

# Define output path for the CSV file
csv_output_path = output_dir / "merged_suricata_alerts.csv"

# Save the DataFrame as a CSV file
merged_alerts_df.to_csv(csv_output_path, index=False)

# === OPTIONAL: Display result (if running interactively) ===
print(f"✅ Merged {len(merged_alerts_df)} alerts from {len(merged_alerts_df['pcap'].unique())} PCAPs.")
print(f"📄 Saved to: {csv_output_path}")
