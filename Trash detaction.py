import cv2
import numpy as np
from twilio.rest import Client
import firebase_admin
from firebase_admin import credentials, firestore
import time
import datetime

# Twilio Client Setup
account_sid = 'ACdf1086f0539977324e2dc952a855f358'
auth_token = 'ffa1631fa5f378cd3d893bf31987759f'
twilio_client = Client(account_sid, auth_token)
twilio_number = '+12792064935'  # Must be SMS & call-enabled
alert_recipient_number = '+918373975610'  # In E.164 format

# Firebase Initialization
cred = credentials.Certificate(r'C:\Users\theda\Documents\phantompulse-firebase-adminsdk-fbsvc-875e865453.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load YOLOv4 Model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load COCO Class Labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO Output Layer Names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Video Capture (0 = Webcam)
cap = cv2.VideoCapture(0)

# Call rate-limiting setup
last_call_time = 0
CALL_DELAY = 30 * 60  # 30 minutes


# Function to send a voice call alert
def make_call_alert():
    global last_call_time
    current_time = time.time()

    if current_time - last_call_time >= CALL_DELAY:
        try:
            call = twilio_client.calls.create(
                twiml='<Response><Say>Trash detected in Area 1. Please clean the area.</Say></Response>',
                from_=twilio_number,
                to=alert_recipient_number
            )
            last_call_time = current_time
            print(f"📞 Call initiated to {alert_recipient_number}")
        except Exception as e:
            print(f"❌ Call failed: {e}")
    else:
        print("⏳ Call skipped (30 min delay not passed)")


# Function to send an SMS alert
def sms_alert():
    try:
        message = twilio_client.messages.create(
            body="🚨 Trash detected in Area 1. Please clean the area immediately. The cordinates of trash area are ()",
            from_=twilio_number,
            to=alert_recipient_number
        )
        print(f"📩 SMS sent to {alert_recipient_number}: SID={message.sid}")
    except Exception as e:
        print(f"❌ SMS failed: {e}")


# Function to log detection to Firebase
def log_bottle_complaint(location, timestamp):
    try:
        doc_ref = db.collection('bottle_detections').document()
        doc_ref.set({
            'location': location,
            'timestamp': timestamp,
            'status': 'Pending'
        })
        print(f"🗃️ Firebase log: {location} at {timestamp}")
    except Exception as e:
        print(f"❌ Firebase log failed: {e}")


# Main Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not found.")
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Detect only bottles (class name = 'bottle')
            if confidence > 0.5 and classes[class_id] == "bottle":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = "Bottle"
            confidence = confidences[i]
            color = (0, 255, 0)

            # Draw detection on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"✅ Trash Detected (Bottle) with {confidence:.2f} confidence.")

            # Alerting and Logging
            make_call_alert()
            sms_alert()

            location = "Detected TRASH at Area 1"
            timestamp = datetime.datetime.now().isoformat()
            log_bottle_complaint(location, timestamp)

    # Show output frame
    cv2.imshow('Bottle Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
