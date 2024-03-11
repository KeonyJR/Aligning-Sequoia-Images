class AlignmentSequoia():
    def __init__(self, ratio=0.5, ReprojThreshold=2):
        # Constructor method for AlignmentSequoia class.
        # Initializes the object with default values for the ratio and reprojection threshold.
        # Attributes:
        # - ratio: Ratio used for Lowe's ratio test during feature matching.
        # - ReprojThreshold: Threshold used for RANSAC during finding homography.

        self.ratio = ratio
        self.ReprojThreshold = ReprojThreshold
        self.num_maches=4  # Number of minimum matches required for alignment.

    def alignment(self, img_source, img_destination, width, height):
        # Method to align two images.
        # Parameters:
        # - img_source: Source image (image to be aligned).
        # - img_destination: Destination image (reference image).
        # - width: Width of the output aligned image.
        # - height: Height of the output aligned image.
        # Returns:
        # - aligned_image: Aligned image.
        # - error_reprojection: Reprojection error.

        # Resize images to the specified width and height
        img_source = cv2.resize(img_source, (width, height), interpolation=cv2.INTER_LINEAR)
        img_destination = cv2.resize(img_destination, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Extract SIFT keypoints and descriptors from source and destination images
        descriptor = cv2.SIFT_create()
        (kpsBase, featuresBase) = descriptor.detectAndCompute(img_source, None)
        (kpsAdditional, featuresAdditional) = descriptor.detectAndCompute(img_destination, None)

        # Match features between source and destination images
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresBase, featuresAdditional, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # If enough matches are found, estimate the homography and align the image
        if len(matches) > self.num_maches:
            ptsA = np.float32([kpsBase[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsAdditional[i].pt for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.ReprojThreshold)
            error_reprojection = self.calculate_reprojection_error(ptsA, ptsB, H)
            print("Reprojection error:", error_reprojection)
            aligned_image = cv2.warpPerspective(img_source, H, (img_source.shape[1], img_source.shape[0]))
        else:
            if matches is None:
                print("Warning: Few matches were found. Try adjusting parameters or preprocessing images.")
            else:
                print("Error: No matches found. The images may be too different or poorly aligned. Consider improving image quality or using a different alignment method.")

        return aligned_image ,error_reprojection
    
    def calculate_reprojection_error(self, pts1, pts2, H):
        # Calculate reprojection error between the original points (pts1) and the transformed points (pts2) using homography H.
        # Parameters:
        # - pts1: Array of shape (n, 2) containing original points.
        # - pts2: Array of shape (n, 2) containing transformed points.
        # - H: Homography matrix.
        # Returns:
        # - error: Mean reprojection error.

        # Transform original points using homography H
        pts1_homog = np.concatenate([pts1, np.ones((len(pts1), 1))], axis=1)
        pts1_transformed = np.dot(H, pts1_homog.T).T
        pts1_transformed /= pts1_transformed[:, 2][:, np.newaxis]
        
        # Calculate Euclidean distance between transformed points and original points
        distances = np.linalg.norm(pts1_transformed[:, :2] - pts2, axis=1)
        
        # Calculate mean reprojection error
        error = np.mean(distances)
        return error
    
    def generate_alignments(self, img_source, img_destination, width, height, num_iterations=3):
        # Generate alignments by varying ratio and ReprojThreshold parameters
        alignments = []
        for _ in range(num_iterations):
            # Perform alignment with current parameters
            aligned_image, error_reprojection = self.alignment(img_source, img_destination, width, height)
            # Store the alignment along with its error
            alignments.append((aligned_image, error_reprojection, self.ratio, self.ReprojThreshold))
            # Update parameters for next iteration
            self.update_parameters()
        # Sort alignments based on reprojection error
        alignments.sort(key=lambda x: x[1])
        # Return the top three alignments
        return alignments[:3]

    def update_parameters(self):
        # Update ratio and ReprojThreshold for the next iteration
        # For simplicity, we'll just decrease ratio by 0.1 
        self.ratio -= 0.01
        # self.ReprojThreshold += 1

#------------------------------USAGE EXAMPLE------------------------------------------
# Import the AlignmentSequoia class
images_alignment = AlignmentSequoia()

# Define the path of the image and coefficients for NDVI calculation
Coef_Red = 1.573522923854364e-05
Coef_NIR = 1.4184047512922233e-05

# Load NIR and RED images
img_NIR = cv2.imread('Data Sample/NIR.TIF')
img_RED = cv2.imread('Data Sample/RED.TIF')

# Convert images to grayscale and adjust by coefficients
im_red_array = cv2.cvtColor(img_RED, cv2.COLOR_BGR2GRAY)
im_NIR_array = cv2.cvtColor(img_NIR, cv2.COLOR_BGR2GRAY)
im_NIR_array = Coef_NIR * np.asarray(im_NIR_array) * 255
im_red_array = Coef_Red * np.asarray(im_red_array) * 255

# Calculate NDVI before aligning the images
NDVI_BEFORE_Allignment = ((im_NIR_array - im_red_array)) / ((im_NIR_array + im_red_array))

# Specify width and height for image alignment
width, height = 1280, 960

# Align RED image with NIR image
aligment_RED, error_reprojection = images_alignment.alignment(img_RED, img_NIR, width, height)

# Convert aligned image to grayscale and adjust by RED coefficient
im_red_array = cv2.cvtColor(aligment_RED, cv2.COLOR_BGR2GRAY)
im_red_array = Coef_Red * np.asarray(im_red_array) * 255

# Calculate NDVI after alignment
NDVI_After_Allignment = ((im_NIR_array - im_red_array)) / ((im_NIR_array + im_red_array))

# Visualize the results before and after alignment
fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
f1 = ax1.imshow(NDVI_BEFORE_Allignment, cmap='RdYlGn', vmin=-1.0, vmax=1.0)
ax1.set_title('NDVI Before Alignment')
f2 = ax2.imshow(NDVI_After_Allignment, cmap='RdYlGn', vmin=-1.0, vmax=1.0)
ax2.set_title('NDVI After Alignment')
ax1.axis('off')
ax2.axis('off')
plt.tight_layout()
plt.show()
