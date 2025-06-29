import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from twilio.rest import Client


ACCOUNT_SID = 'ACdf1086f0539977324e2dc952a855f358'
AUTH_TOKEN = 'ffa1631fa5f378cd3d893bf31987759f'
TWILIO_NUM = '+12792064935'
MY_NUMBER = '+918373975610'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.8], device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_embedding(image_input):
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        img_cropped = img_cropped.unsqueeze(0).to(device)
        embedding = resnet(img_cropped)
        return embedding.squeeze().detach().cpu().numpy()
    return None


def is_match(embedding1, embedding2, threshold=0.6):
    return cosine(embedding1, embedding2) < threshold


def make_call():
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    call = client.calls.create(
        to=MY_NUMBER,
        from_=TWILIO_NUM,
        url='http://demo.twilio.com/docs/voice.xml'
    )
    print(f"Call initiated: {call.sid}")


def recognize_face_in_video(reference_image_path):
    reference_embedding = get_embedding(reference_image_path)
    if reference_embedding is None:
        print("Failed to load reference image.")
        return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(frame)
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for box in boxes:
                face = img.crop((box[0], box[1], box[2], box[3]))
                face_embedding = get_embedding(face)
                if face_embedding is not None and is_match(reference_embedding, face_embedding):
                    print("Face matched! Initiating call...")
                    make_call()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reference_image_path = r'C:\Users\theda\Documents\reference_face.jpg'
    recognize_face_in_video(reference_image_path)
