import cv2
import numpy as np

# Load face detector and landmark model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_model = cv2.face.createFacemarkLBF()
landmark_model.loadModel("lbfmodel.yaml")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    #cv2.imshow('m',img)
   # cv2.waitKey(0)
    
    
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 1:
        print(f"Expected one face in {image_path}, but found {len(faces)}!")
        exit(1)
    (x,y,w,h) = faces[0]
    cropped = img[y:y+h, x:x+w]
    
    resized = cv2.resize(cropped, (400,400))
    
    return resized


def detect_landmarks(img):
    gray = img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    _, landmarks = landmark_model.fit(gray, faces)
    return landmarks[0][0]

# Utility function to apply affine transformation to triangles
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Warping triangles from source to destination
def warp_triangles(img1, img2, tri1, tri2):
        # Rect around the triangle
    r1 = cv2.boundingRect(np.float32([tri1]))
    r2 = cv2.boundingRect(np.float32([tri2]))

    # Offset points by left top corner of the respective rectangles
    tri1_rect = []
    tri2_rect = []
    tri2_rect_int = []

    for i in range(0, 3):
        tri_rect = ((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1]))
        tri1_rect.append(tri_rect)

        tri_rect = ((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1]))
        tri2_rect.append(tri_rect)

        tri2_rect_int.append((int(tri2[i][0] - r2[0]), int(tri2[i][1] - r2[1])))

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = np.zeros((r2[3], r2[2], 3), dtype = img1_rect.dtype)

    
    # Apply affine transform
    img2_rect = apply_affine_transform(img1_rect, tri1_rect, tri2_rect, (r2[2], r2[3]))
    
    # Place the triangle in the output image
    mask = np.zeros_like(img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]])
    cv2.fillConvexPoly(mask, np.int32(tri2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    img2_rect = cv2.resize(img2_rect, (mask.shape[1], mask.shape[0]))
    img2_rect = img2_rect * mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1.0 - mask) + img2_rect
  

# List of images

image_filenames = ["1.JPG","imron.JPG"]

all_landmarks = []
for image_path in image_filenames:
    print(image_path)
    img = preprocess_image(image_path)
    landmarks = detect_landmarks(img)

    all_landmarks.append(landmarks)

# Compute average landmarks
average_landmarks = all_landmarks[1]
sum_img = np.zeros_like(preprocess_image(image_filenames[0]), dtype=np.float64)

# Use float64 for average_img to hold sum of all images
average_img = np.zeros_like(preprocess_image(image_filenames[1]), dtype=np.float64)

# Create a bounding rectangle around the landmarks
rect = (0, 0, 399, 399)
average_landmarks[:, 0] = np.clip(average_landmarks[:, 0], rect[0], rect[2])
average_landmarks[:, 1] = np.clip(average_landmarks[:, 1], rect[1], rect[3])

# Use Delaunay triangulation
epsilon = 0.1
average_landmarks[average_landmarks[:, 0] == 399, 0] -= epsilon
average_landmarks[average_landmarks[:, 1] == 399, 1] -= epsilon
average_landmarks[average_landmarks[:, 0] == 0, 0] += epsilon
average_landmarks[average_landmarks[:, 1] == 0, 1] += epsilon
subdiv = cv2.Subdiv2D(rect)
flat_landmarks = [tuple(pt) for pt in average_landmarks]

# Diagnostic to check for duplicate landmarks
unique_landmarks = set(flat_landmarks)
if len(unique_landmarks) != len(flat_landmarks):
    print("There are duplicate landmarks!")

# Diagnostic to print the landmarks
print(flat_landmarks)

# Insert landmarks one-by-one and print the one that fails
for landmark in flat_landmarks:
    try:
        subdiv.insert(landmark)
    except:
        print(f"Failed to insert landmark: {landmark}")


triangles_list = subdiv.getTriangleList()


# Warp all images to average landmarks
for i, image_path in enumerate(image_filenames[:2]):
    print(i, image_path)
    img = preprocess_image(image_path)
    cv2.imshow(image_path,img)

for i, image_path in enumerate(image_filenames[:1]):

    img = preprocess_image(image_path)


    warped_img = np.zeros_like(preprocess_image(image_filenames[1]), dtype=np.float64)


    for t in triangles_list:
        # Retrieve the landmark indices based on coordinates (use your landmark detection to figure out the indices)
        pt1 = np.where((average_landmarks == (t[0], t[1])).all(axis=1))[0]
        pt2 = np.where((average_landmarks == (t[2], t[3])).all(axis=1))[0]
        pt3 = np.where((average_landmarks == (t[4], t[5])).all(axis=1))[0]

        # Ensure the indices are found
        if len(pt1) == 0 or len(pt2) == 0 or len(pt3) == 0:
            continue

        x, y, z = pt1[0], pt2[0], pt3[0]

        tri1 = [all_landmarks[i][x], all_landmarks[i][y], all_landmarks[i][z]]
        tri2 = [average_landmarks[x], average_landmarks[y], average_landmarks[z]]
        warp_triangles(img, warped_img, tri1, tri2)

    sum_img = warped_img

# Divide by the number of images to get average
average_img = sum_img
average_img = np.uint8(average_img)

cv2.imshow('Average Face', average_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
