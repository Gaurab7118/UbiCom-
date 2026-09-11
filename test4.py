import pandas as pd
import ollama
import os

# Clear terminal for a clean interface
os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_CSV = r"D:\UbiCom Data\Data\gaurab_upload\predicted_full_format.csv"
LABELS = [
    'label:LYING_DOWN', 'label:SITTING', 'label:STANDING_IN_PLACE',
    'label:STANDING_AND_MOVING', 'label:WALKING', 'label:RUNNING', 'label:BICYCLING'
]

# ==========================================
# 2. AGGREGATE THE CSV INTO A TEXT TIMELINE
# ==========================================
def generate_timeline_text(csv_path):
    print("Reading CSV and compressing timeline for the SLM...")
    df = pd.read_csv(csv_path)
    
    timeline_text = "User Activity Timeline:\n"
    current_activity = None
    start_time = None
    
    for index, row in df.iterrows():
        active_label = "UNKNOWN"
        for label in LABELS:
            if row[label] == 1:
                active_label = label.replace("label:", "")
                break
                
        if active_label != current_activity:
            if current_activity is not None:
                timeline_text += f"Timestamp(s): {start_time:.0f} to {row['time_offset']:.0f} -> {current_activity}\n"
            current_activity = active_label
            start_time = row['time_offset']
            
    timeline_text += f"Timestamp(s): {start_time:.0f} to {df.iloc[-1]['time_offset']:.0f} -> {current_activity}\n"
    
    return timeline_text

# ==========================================
# 3. QUERY THE LOCAL PHI-3 MODEL
# ==========================================
def ask_phi3(timeline_context, user_question):
    # We force the model to use your exact structure
    full_prompt = f"""
    You are a strict data analysis engine. Analyze the following activity timeline:
    
    {timeline_context}
    
    You MUST answer the user's query by filling out the exact template below. Do not add any conversational text before or after the template.
    If the question asks for a specific time, calculate it from the timeline.
    
    TEMPLATE:
    Query: "{user_question}"
    Answer: [Provide a concise direct answer]
    Activity/Event: [Name of the primary activities involved]
    Evidence:
        Timestamp(s): [List the exact seconds from the timeline, e.g., 905 to 1420. If none, write N/A]
        Sensor Modality: Accelerometer, Gyroscope
        Sensor Channel(s): All
    Explanation: [Write exactly one detailed sentence explaining how the timeline data proves your answer.]
    """
    
    response = ollama.chat(model='qwen2.5:1.5b', messages=[
        {'role': 'user', 'content': full_prompt}
    ])
    
    print("\n" + response['message']['content'] + "\n")

# ==========================================
# 4. INTERACTIVE TERMINAL LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Compress the data only ONCE when the script starts
    compressed_timeline = generate_timeline_text(INPUT_CSV)
    
    print("\n" + "="*60)
    print("✅ SYSTEM READY. The SLM has read the sensor data.")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("="*60)
    
    # 2. Start the interactive loop
    while True:
        question = input("\nQuery: ")
        
        # Check if the user wants to close the program
        if question.lower() in ['exit', 'quit', 'q']:
            print("Closing the system.")
            break
            
        # Skip if the user just pressed Enter without typing anything
        if question.strip() == "":
            continue
            
        print("Analyzing...")
        ask_phi3(compressed_timeline, question)
        print("-" * 60)