import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import cv2
import dlib
import imutils
from imutils import face_utils
import time

def localize_face_from_image_path(img_path, verbose = False):
    if verbose:
        print(img_path)
    
    img = cv2.imread(img_path)
    return localize_face(img, verbose)

def localize_face(img, display_result = True, verbose = False):    
    if verbose:
        cv2.imshow('Original', img)
        cv2.waitKey(0)
    
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    
    faces = list()
    for i, rect in enumerate(rects):
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.putText(img, f'Face:{i+1}', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

        face = img[y:y+h, x:x+w]
        if verbose:
            cv2.imshow('Face', face)
            cv2.waitKey(0)
        faces.append(face)

    if display_result:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return faces

def load_init_model(verbose = False):
    model = torchvision.models.vgg16_bn(pretrained=True)

    for params in model.parameters():
        params.requires_grad = False
    
    if verbose:
        print(model)

    in_features = model.classifier[6].in_features
    layers = list(model.classifier.children())[:-1]
    layers.extend([nn.Linear(in_features, 2)])
    model.classifier = nn.Sequential(*layers)

    return model

def process_image(img_path, model, transformation, classes, verbose=False):
    faces = localize_face_from_image_path(img_path, verbose)
    for i, face in enumerate(faces):
        start = time.time()
        face_tensor = transformation(face).float()
        face_tensor = Variable(face_tensor, requires_grad=True)
        face_tensor = face_tensor.unsqueeze(0) # this is for VGG, others might not require it
        print(face_tensor.size())
        
        output = model(face_tensor)
        _, preds = torch.max(output, dim=1)

        end = time.time()

        print(classes[preds.item()])
        print(f'Prediction time for Face:{i+1} = {end-start} sec')

def process_video_using_webcam(model, transformation, classes, verbose=False):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        
        # frame = imutils.resize(frame, width=500)
        # faces = localize_face(frame, display_result=False, verbose=verbose)
        # for i, face in enumerate(faces):
        #     start = time.time()
        #     face_tensor = transformation(face).float()
        #     face_tensor = Variable(face_tensor, requires_grad=True)
        #     face_tensor = face_tensor.unsqueeze(0) # this is for VGG, others might not require it
            
        #     output = model(face_tensor)
        #     _, preds = torch.max(output, dim=1)

        #     end = time.time()

        #     # print(classes[preds.item()])
        #     print(f'Prediction time for Face:{i+1} = {end-start} sec')

        #     cv2.putText(frame, f'Face({i+1}):{classes[preds.item()]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        start = time.time()
        face_tensor = transformation(frame).float()
        face_tensor = Variable(face_tensor, requires_grad=True)
        face_tensor = face_tensor.unsqueeze(0) # this is for VGG, others might not require it
        
        output = model(face_tensor)
        _, preds = torch.max(output, dim=1)

        end = time.time()

        i = 0
        # print(classes[preds.item()])
        print(f'Prediction time for Face:{i+1} = {end-start} sec')

        cv2.putText(frame, f'Face({i+1}):{classes[preds.item()]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = '.\\data\\test_images\\1.jpg'
    model_path = '.\\models\\mask_detector.pt'
    classes = ['with_mask', 'without_mask']
    verbose = True
    
    model = load_init_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)

    transformation = transforms.Compose([transforms.ToPILImage(), transforms.Resize((300, 400)), transforms.ToTensor()])
    # process_image(img_path, model, transformation, classes, verbose)
    process_video_using_webcam(model, transformation, classes, verbose=False)



