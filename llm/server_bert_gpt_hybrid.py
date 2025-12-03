# server_bert_gpt_hybrid.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import fastapi_poe as fp
import asyncio

app = FastAPI()

POE_API_KEY = "aA_SPfposL5Zgrm3qft9ufaalxrjpkZdvElonE2lG4w"

# Load BioBERT model for embeddings
print("Loading BioBERT model for embeddings...")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

print(f"BioBERT loaded on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Load training data
print("Loading training data...")
try:
    df = pd.read_excel("/mnt/user-data/outputs/medical_device_training_data.xlsx")
    print(f"Loaded {len(df)} training samples")
except:
    print("Warning: Training data not found.")
    df = None

class DeviceData(BaseModel):
    error_code: str = None
    device_type: str
    temperature: float
    additional_info: str = ""
    device_id: str = "UNKNOWN"

def embed_text(text):
    """Generate BioBERT embedding for text"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    return vec

def create_training_embeddings():
    """Create embeddings for all training samples"""
    if df is None or len(df) == 0:
        return None, None
    
    print("Creating embeddings for training data...")
    
    training_texts = []
    for _, row in df.iterrows():
        text = f"""Error Code: {row['error_code']}
Device Type: {row['device_type']}
Temperature: {row['temperature']}°C
Additional Info: {row['additional_info']}"""
        training_texts.append(text)
    
    embeddings = []
    for text in tqdm(training_texts, desc="Creating embeddings"):
        embeddings.append(embed_text(text))
    
    return np.vstack(embeddings), training_texts

# Create embeddings at startup
TRAINING_EMBEDDINGS, TRAINING_TEXTS = create_training_embeddings()

def find_similar_cases(query_embedding, top_k=3):
    """Find most similar cases from training data"""
    if TRAINING_EMBEDDINGS is None:
        return []
    
    query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, TRAINING_EMBEDDINGS)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    similar_cases = []
    for idx in top_indices:
        similar_cases.append({
            "similarity": float(similarities[idx]),
            "error_code": df.iloc[idx]['error_code'],
            "device_type": df.iloc[idx]['device_type'],
            "temperature": df.iloc[idx]['temperature'],
            "severity": df.iloc[idx]['severity_level'],
            "fda_error_code": df.iloc[idx]['fda_error_code'],
            "diagnosis": df.iloc[idx]['diagnostic_hypothesis'],
            "troubleshooting": df.iloc[idx]['troubleshooting_steps'],
            "additional_info": df.iloc[idx]['additional_info']
        })
    
    return similar_cases

async def get_gpt_diagnosis(device_data, similar_cases):
    """
    Use GPT-4o-mini to generate diagnosis based on device data and similar cases
    This is the hybrid approach: BERT finds context, GPT generates diagnosis
    """
    
    # Build context from similar cases
    similar_context = "\n\n**Context from Similar Historical Cases:**\n"
    for i, case in enumerate(similar_cases, 1):
        similar_context += f"""
Case {i} (Similarity: {case['similarity']:.1%}):
- Error: {case['error_code']}
- Device: {case['device_type']}
- Temp: {case['temperature']}°C
- Severity: {case['severity']}
- FDA Code: {case['fda_error_code']}
- Diagnosis: {case['diagnosis'][:150]}...
"""
    
    # Create prompt for GPT
    prompt = f"""You are a medical device diagnostic assistant. Analyze this new case using the similar historical cases as context.

**NEW CASE:**
Device Type: {device_data.device_type}
Temperature: {device_data.temperature}°C
Additional Info: {device_data.additional_info}
Device ID: {device_data.device_id}
{similar_context}

Based on the similar cases above, provide a diagnosis following this format:

**FDA Error Code**: [Classification]
**Severity Level**: [Critical/High/Medium/Low]
**Device Status**: [Status]

**Diagnostic Hypothesis**:
- Primary cause: [explanation]
- Contributing factors: [if any]
- Confidence level: [percentage based on similarity to past cases]

**Troubleshooting Steps**:
[Numbered steps based on similar cases]

**Additional Notes**: [Observations]

Use the similar cases to inform your diagnosis, but adapt to this specific situation.
"""

    messages = [
        fp.ProtocolMessage(role="user", content=prompt)
    ]
    
    full_response = ""
    
    try:
        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name="GPT-4o-Mini",
            api_key=POE_API_KEY
        ):
            full_response += partial.text
        
        return full_response
    
    except Exception as e:
        return f"Error communicating with GPT: {str(e)}"

@app.post("/diagnose")
async def diagnose_device(data: DeviceData):
    """
    Hybrid approach:
    1. Use BioBERT to find similar cases (fast retrieval)
    2. Feed similar cases to GPT-4o-mini for enhanced diagnosis
    """
    
    print(f"\nHybrid Diagnosis for device: {data.device_id}")
    print(f"   Device: {data.device_type}")
    print(f"   Temp: {data.temperature}°C")
    
    try:
        # Step 1: BioBERT - Find similar cases
        print("   [1/2] BioBERT: Finding similar cases...")
        query_text = f"""Device Type: {data.device_type}
Temperature: {data.temperature}°C
Additional Info: {data.additional_info}"""
        
        query_embedding = embed_text(query_text)
        similar_cases = find_similar_cases(query_embedding, top_k=3)
        
        print(f"   Found {len(similar_cases)} similar cases")
        for i, case in enumerate(similar_cases, 1):
            print(f"      {i}. {case['error_code']} (similarity: {case['similarity']:.2%})")
        
        # Step 2: GPT-4o-mini - Generate enhanced diagnosis
        print("   [2/2] GPT-4o-mini: Generating diagnosis...")
        diagnosis = await get_gpt_diagnosis(data, similar_cases)
        
        print(f"   Diagnosis complete")
        
        return {
            "status": "success",
            "device_id": data.device_id,
            "diagnosis": diagnosis,
            "timestamp": datetime.now().isoformat(),
            "model": "BioBERT → GPT-4o-mini (Hybrid)",
            "similar_cases": similar_cases,
            "approach": "hybrid",
            "step_1": "BioBERT embeddings for similarity search",
            "step_2": "GPT-4o-mini for context-aware diagnosis"
        }
    
    except Exception as e:
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Medical Device Diagnostic Server - BERT → GPT Hybrid",
        "status": "online",
        "approach": "BioBERT embeddings + GPT-4o-mini",
        "training_samples": len(df) if df is not None else 0,
        "step_1": "BioBERT finds similar cases",
        "step_2": "GPT generates diagnosis with context"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "Hybrid"}

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Hybrid Diagnostic Server...")
    print("Architecture: BioBERT (similarity) → GPT-4o-mini (diagnosis)")
    print(f"Training samples: {len(df) if df is not None else 0}")
    print("Listening on http://0.0.0.0:5004\n")
    uvicorn.run(app, host="0.0.0.0", port=5004)
