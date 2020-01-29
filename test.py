import nibabel
import scipy.optimize
import sirf.Reg as reg
import os
import numpy as np
import re


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    return [atoi(c) for c in re.split(r'(\d+)', string)]


def warp_image_forward(resampler, static_image):
    return resampler.forward(static_image).as_array()


def warp_image_adjoint(resampler, dynamic_image):
    return resampler.adjoint(dynamic_image).as_array()


def edit_header(data, output_path):
    new_data = []

    for i in range(len(data)):
        current_data = nibabel.load(data[i])
        current_data.header["intent_code"] = 1007

        current_data_path = "{0}/new_dvf_{1}.nii".format(output_path, str(i))

        nibabel.save(current_data, current_data_path)

        new_data.append(current_data_path)

    return new_data


def objective_function(static_array, resampler, dynamic_images, static_image):
    static_image.fill(static_array)

    objective_value = 0.0

    for i in range(len(dynamic_images)):
        objective_value = objective_value + (np.nansum(np.square(dynamic_images[i].as_array() - warp_image_forward(resampler[i], static_image)), dtype=np.double) / 2.0)

    print(objective_value)

    return objective_value


def gradient_function(static_array, resampler, dynamic_images, static_image):
    static_image.fill(static_array)

    gradient_value = static_image.clone()
    gradient_value.fill(0.0)

    adjoint_image = static_image.clone()

    for i in range(len(dynamic_images)):
        adjoint_image.fill(warp_image_forward(resampler[i], static_image) - dynamic_images[i].as_array())

        gradient_value.fill(gradient_value.as_array() + warp_image_adjoint(resampler[i], adjoint_image))

    gradient_value.write("/home/sirfuser/Shared_Folder/gradient.nii")

    print(gradient_value.as_array().max())

    return gradient_value.as_array()


def get_data_path(data_path):
    all_input_data = os.listdir(data_path)
    input_data = []

    for i in range(len(all_input_data)):
        current_input_data = all_input_data[i].rstrip()

        if len(current_input_data.split(".nii")) > 1 and len(current_input_data.split("fixed")) > 1:
            input_data.append("{0}/{1}".format(data_path, current_input_data))

    input_data.sort(key=human_sorting)

    return input_data


def get_dvf_path(dvf_path):
    all_dvf_data = os.listdir(dvf_path)
    dvf_data = []

    for i in range(len(all_dvf_data)):
        current_dvf_data = all_dvf_data[i].rstrip()

        if len(current_dvf_data.split("DVF")) > 1:
            dvf_data.append("{0}/{1}".format(dvf_path, current_dvf_data))

    dvf_data.sort(key=human_sorting)

    return dvf_data


def main():
    data_path = '/home/sirfuser/Shared_Folder/cropped_input/dynamic'
    dvf_path = '/home/sirfuser/Shared_Folder/D_fields'
    new_dvf_path = '/home/sirfuser/Shared_Folder/D_fields_new'

    input_data = get_data_path(data_path)
    dvf_data = get_dvf_path(dvf_path)

    input_array = []

    for i in range(len(input_data)):
        input_array.append(reg.NiftiImageData(input_data[i]))

    dvf_data = edit_header(dvf_data, new_dvf_path)

    dvf_array = []

    for i in range(len(dvf_data)):
        dvf_array.append(reg.NiftiImageData3DDeformation(dvf_data[i]))

    static_image = input_array[0].clone()

    resamplers = []

    for i in range(len(input_array)):
        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(input_array[i])
        resampler.add_transformation(dvf_array[i])

        resampler.set_interpolation_type_to_cubic_spline()

        resamplers.append(resampler)

    input_resampler = [resamplers[0]]
    input_input_data = [input_array[0]]

    static_array = static_image.as_array()

    warp = warp_image_forward(resamplers[0], static_image)

    warped_image = static_image.clone()
    warped_image.fill(warp)

    warped_image.write("/home/sirfuser/Shared_Folder/warp_forward.nii")

    difference = static_image.as_array() - warp

    difference_image = static_image.clone()
    difference_image.fill(difference)

    difference_image.write("/home/sirfuser/Shared_Folder/difference.nii")

    for i in range(len(input_array)):
        warp = warp_image_adjoint(resamplers[i], input_array[i])

        warped_image = input_array[i].clone()
        warped_image.fill(warp)

        warped_image.write("/home/sirfuser/Shared_Folder/warp_adjoint_{0}.nii".format(str(i)))

    scipy.optimize.minimize(objective_function, static_array, args=(input_resampler, input_input_data, static_image), method="L-BFGS-B", jac=gradient_function)


main()
