from flask import Flask, request
from twilio.rest import Client
from dotenv import load_dotenv
import os
from llm_genration import process_query, load_langchain_vectorstore, load_disease_csv

# CONFIG

load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
account_sid = TWILIO_ACCOUNT_SID  
auth_token = TWILIO_AUTH_TOKEN   


twilio_whatsapp_number = 'whatsapp:+14155238886'


client = Client(account_sid, auth_token)

# Initialize Flask + load models

app = Flask(__name__)

db, embedding_model = load_langchain_vectorstore()
disease_df = load_disease_csv("data/disease_symptoms.csv")

# WhatsApp Webhook

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    # Read incoming WhatsApp message
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "").strip()

    print(f"Incoming WhatsApp message from {from_number}: {incoming_msg}")

    if not incoming_msg:
        return "Missing message content.", 400

    try:
        if incoming_msg.lower() in ("quit", "exit"):
            reply = "Goodbye! Stay healthy. 👋"
        else:
            # Call your LLM pipeline
            reply = process_query(
                incoming_msg,
                disease_df,
                db,
                embedding_model,
                return_response=True
            )
    except Exception as e:
        print(f"Error: {e}")
        reply = "Sorry, an error occurred while processing your request."

    # Send reply via Twilio REST API

    try:
        message = client.messages.create(
            body=reply,
            from_=twilio_whatsapp_number,
            to=from_number
        )
        print("Twilio message SID:", message.sid)
    except Exception as e:
        print(f"Twilio sending error: {e}")
        return "Error sending reply via Twilio.", 500

    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
