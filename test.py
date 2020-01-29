import nibabel
import scipy.optimize
import sirf.Reg as reg
import sirf.Reg as eng_ref
import sirf.Reg as eng_flo
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
    static_image.fill(np.reshape(static_array, static_image.as_array().shape))

    objective_value = 0.0

    for i in range(len(dynamic_images)):
        objective_value = objective_value + (np.nansum(np.square(dynamic_images[i].as_array() - warp_image_forward(resampler[i], static_image)), dtype=np.double) / 2.0)

    print("Objective function value: {0}".format(str(objective_value)))

    return objective_value


def gradient_function(static_array, resampler, dynamic_images, static_image):
    static_image.fill(np.reshape(static_array, static_image.as_array().shape))

    gradient_value = static_image.clone()
    gradient_value.fill(0.0)

    adjoint_image = static_image.clone()

    for i in range(len(dynamic_images)):
        static_image.write("/home/sirfuser/Shared_Folder/temp_static.nii")
        dynamic_images[i].write("/home/sirfuser/Shared_Folder/temp_dynamic.nii")

        temp_static = reg.NiftiImageData("/home/sirfuser/Shared_Folder/temp_static.nii")
        temp_dynamic = reg.NiftiImageData("/home/sirfuser/Shared_Folder/temp_dynamic.nii")

        adjoint_image.fill(warp_image_forward(resampler[i], temp_static) - temp_dynamic.as_array())

        gradient_value.fill(gradient_value.as_array() + warp_image_adjoint(resampler[i], adjoint_image))

    gradient_value.write("/home/sirfuser/Shared_Folder/gradient.nii")

    print("Max gradient value: {0}, Min gradient value: {1}, Mean gradient value: {2}, Gradient norm: {3}".format(str(gradient_value.as_array().max()), str(gradient_value.as_array().min()), str(np.nanmean(gradient_value.as_array())), str(np.linalg.norm(gradient_value.as_array()))))

    return np.ravel(gradient_value.as_array())


def register_data(ref_file, input_data):
    path_new_displacementfields = '/home/sirfuser/Shared_Folder/new_displacement_fields/'
    if not os.path.exists(path_new_displacementfields):
        os.makedirs(path_new_displacementfields, mode=0o770)

    path_new_deformationfields = '/home/sirfuser/Shared_Folder/new_deformation_fields/'
    if not os.path.exists(path_new_deformationfields):
        os.makedirs(path_new_deformationfields, mode=0o770)

    path_new_tm = '/home/sirfuser/Shared_Folder/new_tm/'
    if not os.path.exists(path_new_tm):
        os.makedirs(path_new_tm, mode=0o770)

    algo = reg.NiftyAladinSym()

    dvf_array = []

    for i in range(len(input_data)):
        flo_file = input_data[i]
        ref = eng_ref.ImageData(ref_file)
        flo = eng_flo.ImageData(flo_file)

        algo.set_reference_image(ref)
        algo.set_floating_image(flo)

        algo.process()

        output = algo.get_displacement_field_forward()
        output.write('{0}new_displacement_field{1}.nii'.format(path_new_displacementfields, str(i)))

        dvf_array.append('{0}new_DVF_field{1}.nii'.format(path_new_deformationfields, str(i)))

        output2 = algo.get_deformation_field_forward()
        output2.write(dvf_array[i])

        tm = algo.get_transformation_matrix_forward()
        tm.write('{0}new_tm{1}.nii'.format(path_new_tm, str(i)))

    return dvf_array


def test_for_adj(static_image, input_array, dvf_array):
    for i in range(len(input_array)):
        static_image.write("/home/sirfuser/Shared_Folder/temp_static.nii")
        input_array[i].write("/home/sirfuser/Shared_Folder/temp_dynamic.nii")
        dvf_array[i].write("/home/sirfuser/Shared_Folder/temp_dvf.nii")

        temp_static = reg.NiftiImageData("/home/sirfuser/Shared_Folder/temp_static.nii")
        temp_dynamic = reg.NiftiImageData("/home/sirfuser/Shared_Folder/temp_dynamic.nii")
        temp_dvf = reg.NiftiImageData3DDeformation("/home/sirfuser/Shared_Folder/temp_dvf.nii")

        resampler = reg.NiftyResample()

        resampler.set_reference_image(temp_static)
        resampler.set_floating_image(temp_static)
        resampler.add_transformation(temp_dvf)

        resampler.set_interpolation_type_to_linear()

        warp = warp_image_forward(resampler, temp_static)

        warped_image = static_image.clone()
        warped_image.fill(warp)

        warped_image.write("/home/sirfuser/Shared_Folder/warp_forward_{0}.nii".format(str(i)))

        difference = temp_static.as_array() - warp

        difference_image = temp_static.clone()
        difference_image.fill(difference)

        difference_image.write("/home/sirfuser/Shared_Folder/warp_forward_difference_{0}.nii".format(str(i)))

        warp = warp_image_adjoint(resampler, temp_static)

        warped_image = temp_dynamic.clone()
        warped_image.fill(warp)

        warped_image.write("/home/sirfuser/Shared_Folder/warp_adjoint_{0}.nii".format(str(i)))

        difference = temp_static.as_array() - warp

        difference_image = temp_static.clone()
        difference_image.fill(difference)

        difference_image.write("/home/sirfuser/Shared_Folder/warp_adjoint_difference_{0}.nii".format(str(i)))

    return True


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
    # file paths to data
    data_path = '/home/sirfuser/Shared_Folder/cropped_input/dynamic'
    dvf_path = '/home/sirfuser/Shared_Folder/new_deformation_fields'
    new_dvf_path = '/home/sirfuser/Shared_Folder/D_fields_new'

    # get static and dynamic paths
    input_data = get_data_path(data_path)
    static_data = input_data[0]

    # if do reg the calc dvf if not load
    do_reg = False

    if do_reg:
        dvf_data = register_data(static_data, input_data)
    else:
        dvf_data = get_dvf_path(dvf_path)

    # load dynamic objects
    input_array = []

    for i in range(len(input_data)):
        input_array.append(reg.NiftiImageData(input_data[i]))

    # load static objects
    static_image = reg.NiftiImageData(static_data)

    # fix dvf header and load dvf objects
    dvf_data = edit_header(dvf_data, new_dvf_path)

    dvf_array = []

    for i in range(len(dvf_data)):
        dvf_array.append(reg.NiftiImageData3DDeformation(dvf_data[i]))

    # create object to get forward and adj
    resamplers = []

    for i in range(len(input_array)):
        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(input_array[i])
        resampler.add_transformation(dvf_array[i])

        resampler.set_interpolation_type_to_cubic_spline()

        resamplers.append(resampler)

    # test for adj
    do_test_for_adj = False

    if do_test_for_adj:
        test_for_adj(static_image, input_array, dvf_array)

    # array to optimise
    static_array = static_image.as_array()

    # optimise
    static_array = np.reshape(scipy.optimize.minimize(objective_function, np.ravel(static_array), args=(resamplers, input_array, static_image), method="CG", jac=gradient_function).x, static_array.shape, options={"disp": True, "maxiter": -10, "gtol": 1.0})

    # output
    static_image.fill(static_array)
    static_image.write("/home/sirfuser/Shared_Folder/OUTPUT.nii")

main()
