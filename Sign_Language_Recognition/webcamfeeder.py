import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("proj/sign_language_model.keras")
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}
camera = cv2.VideoCapture(0)
message=""
while True:
    ret, frame = camera.read()
    if not ret:
        break
    h,w,_=frame.shape
    x1,y1=int(w*0.25),int(h*0.25)
    x2,y2=int(w*0.75),int(h*0.75)
    cut=frame[y1:y2,x1:x2]
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    input_data = normalized.reshape(1, 28, 28)
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions)
    predicted_letter = label_to_letter.get(predicted_label, "Unknown")
    if(predicted_letter not in label_to_letter.values()):
        message+="J"
    else:
        message+=predicted_letter
    cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Message: {message}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Final Message:", message)
with open("message.txt", "w") as f:
    f.write(message)
camera.release()