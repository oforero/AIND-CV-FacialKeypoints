import cv2
import numpy as np
import keras

def detect_faces(image, draw_box=True):
    # Convert the RGB  image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract the pre-trained face detector from an xml file
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

    # Detect the faces in image
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # Make a copy of the orginal image to draw face detections on
    image_with_detections = np.copy(image)

    # Get the bounding box for each detected face
    if draw_box:
        for (x,y,w,h) in faces:
            # Add a red bounding box to the detections image
            cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)

    return image_with_detections, gray, faces


def detect_eyes(img_color, img_gray, faces):
    # Create an eye detector
    eye_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')

    for (x,y,w,h) in faces:
        cv2.rectangle(img_color, (x,y), (x+w,y+h),(255,0,0), 3)  
        face_gray = img_gray[y:y+h, x:x+w]
        face_color = img_color[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
    return img_color


def blur_faces(img_color, faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(img_color, (x,y), (x+w,y+h),(255,0,0), 3)  
        face_color = img_color[y:y+h, x:x+w]
        
        # Crreate mean blur kernel
        kw, kh, _ = face_color.shape
        kw //= 10 
        kh //= 10
        kernel = np.ones((kw, kh), np.float32)/(kw * kh)
        
        # Apply the kernel
        img_color[y:y+h, x:x+w] = cv2.filter2D(face_color, -1, kernel)
    
    return img_color


def detect_faces_and_eyes(image):
    return detect_eyes(*detect_faces(image))


def blur_faces_in_image(image):
    img_color, _, faces = detect_faces(image)
    return blur_faces(img_color, faces)


def transform_coords_fn(w, h):
    w2, h2 = w // 2, h // 2
    def transform(x, y):
        return int(x * w2 + w2), int(y * h2 + h2)
    return transform


def detect_face_markers_cnn(model, img_color, img_gray, faces):
    for (x,y,w,h) in faces:
        face_gray = img_gray[y:y+h, x:x+w]
        w, h = face_gray.shape
        to_face_coord = transform_coords_fn(w, h)

        face_pred = cv2.resize(face_gray, (96, 96), interpolation = cv2.INTER_CUBIC)
        face_pred = face_pred / 255.
        face_pred = face_pred.reshape((1, 96, 96, 1))
        face_color = img_color[y:y+h, x:x+w]
    
        face_markers = model.predict(face_pred).squeeze(axis=0)
        face_markers = zip(face_markers[0::2], face_markers[1::2])
        for x1, y1 in face_markers:
            # Could this be done better?
            centre = to_face_coord(x1, y1)
            cv2.circle(face_color, centre, 2, (0,255,0), -1)
    
    return img_color


def load_model(model_file):
    return keras.models.load_model(model_file)


def detect_faces_and_markers(model, image):
    return detect_face_markers_cnn(model, *detect_faces(image))


def put_glasses(face, glasses, face_markers, left=9, right=7, offset=15, glasses_to_gray=False):
    to_face_coords = transform_coords_fn(*face.shape[0:2])
    lx, ly = to_face_coords(*face_markers[left])
    rx, ry = to_face_coords(*face_markers[right])

    lx -= offset
    ly -= offset
    rx += offset
    ry += offset
    sh, sw, _ = glasses.shape
    by = float(rx - lx) / sw
    bw, bh = int(sw * by), int(sh * by)

    glasses_rz = cv2.resize(glasses, (bw, bh), interpolation = cv2.INTER_CUBIC)
    alpha_rz = glasses_rz[:, :, 3].reshape(bh, bw, 1) // 255
    glasses_rz = glasses_rz[:, :, 0:3]
    if glasses_to_gray:
        glasses_rz = cv2.cvtColor(glasses_rz, cv2.COLOR_RGB2GRAY) / 255.
        glasses_rz = glasses_rz.reshape(bh, bw, 1)

    face[ly:ly+bh, lx:lx+bw] = face[ly:ly+bh, lx:lx+bw] * (1 - alpha_rz) + glasses_rz * alpha_rz


def detect_faces_add_glasses_cnn(model, img_color, img_gray, faces):
    sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)

    for (x,y,w,h) in faces:
        face_gray = img_gray[y:y+h, x:x+w]
        w, h = face_gray.shape
        to_face_coord = transform_coords_fn(w, h)

        face_pred = cv2.resize(face_gray, (96, 96), interpolation = cv2.INTER_CUBIC)
        face_pred = face_pred / 255.
        face_pred = face_pred.reshape((1, 96, 96, 1))
        face_color = img_color[y:y+h, x:x+w]
    
        face_markers = model.predict(face_pred).squeeze(axis=0)
        face_markers = list(zip(face_markers[0::2], face_markers[1::2]))

        put_glasses(face_color, sunglasses, face_markers)

    return img_color


def detect_faces_add_glasses(model, image, draw_box=True):
    return detect_faces_add_glasses_cnn(model, *detect_faces(image, draw_box))


def process_picture(in_file, out_file, effect):
    img = cv2.imread(in_file)
    model = load_model("my_model.h5")
    #out_img = detect_faces_and_eyes(img)
    out_img = detect_faces_add_glasses(model, img)
    cv2.imwrite(out_file, out_img)


def main(argv):
    if len(argv) < 3:
        print("invalid number of arguments")
    
    if len(argv) == 3:
        argv.append('MARKERS')
    
    _, in_file, out_file, effect = argv

    process_picture(in_file, out_file, effect)

if __name__ == '__main__':
    import sys

    main(sys.argv)
