from skimage.measure import compare_ssim
def ssim(generated_images,y_train):
    sum=0
    for i in range(len(generated_images)):
        s = compare_ssim(generated_images[i,:,:,0],y_train[i,:,:,0],multichannel=True)
        sum = s+sum
    s = sum/len(generated_images)
    return s
