# Copyright University College London 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# Author: Eric Einspänner
# For internal research only.


import os
import sys
import distutils.util
import re
import numpy as np
import scipy.optimize
import nibabel

import sirf.Reg as reg
import sirf.Reg as eng_ref
import sirf.Reg as eng_flo

import parser


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    return [atoi(c) for c in re.split(r'(\d+)', string)]


def warp_image_forward(resampler, static_image):
    return resampler.forward(static_image).as_array().astype(np.double)


def warp_image_adjoint(resampler, dynamic_image):
    return resampler.adjoint(dynamic_image).as_array().astype(np.double)


def objective_function(optimise_array, resampler, dynamic_images, static_image, output_path):
    static_image.fill(np.reshape(optimise_array, static_image.as_array().astype(np.double).shape))

    objective_value = 0.0

    for i in range(len(dynamic_images)):
        objective_value = objective_value + (np.nansum(
            np.square(dynamic_images[i].as_array().astype(np.double) - warp_image_forward(resampler[i], static_image),
                      dtype=np.double), dtype=np.double) / 2.0)

    print("Objective function value: {0}".format(str(objective_value)))

    return objective_value


def gradient_function(optimise_array, resampler, dynamic_images, static_image, output_path):
    static_image.fill(np.reshape(optimise_array, static_image.as_array().astype(np.double).shape))

    gradient_value = static_image.clone()
    gradient_value.fill(0.0)

    adjoint_image = static_image.clone()

    for i in range(len(dynamic_images)):
        static_image.write("{0}/temp_static.nii".format(output_path))
        dynamic_images[i].write("{0}/temp_dynamic.nii".format(output_path))

        temp_static = reg.NiftiImageData("{0}/temp_static.nii".format(output_path))
        temp_dynamic = reg.NiftiImageData("{0}/temp_dynamic.nii".format(output_path))

        adjoint_image.fill(warp_image_forward(resampler[i], temp_static) - temp_dynamic.as_array().astype(np.double))

        gradient_value.fill(
            (gradient_value.as_array().astype(np.double) + warp_image_adjoint(resampler[i], adjoint_image)))

    gradient_value.write("{0}/gradient.nii".format(output_path))

    print("Max gradient value: {0}, Min gradient value: {1}, Mean gradient value: {2}, Gradient norm: {3}".format(
        str(gradient_value.as_array().astype(np.double).max()), str(gradient_value.as_array().astype(np.double).min()),
        str(np.nanmean(gradient_value.as_array().astype(np.double), dtype=np.double)),
        str(np.linalg.norm(gradient_value.as_array().astype(np.double)))))

    return np.ravel(gradient_value.as_array().astype(np.double)).astype(np.double)


def output_input(static_image, dynamic_array, dvf_array, output_path):
    static_image.write("{0}/static_image.nii".format(output_path))

    for i in range(len(dynamic_array)):
        dynamic_array[i].write("{0}/dynamic_array_{1}.nii".format(output_path, str(i)))
        dvf_array[i].write("{0}/dvf_array_{1}.nii".format(output_path, str(i)))

    return True


def test_for_adj(static_image, dvf_array, output_path):
    static_image_path = "{0}/temp_static.nii".format(output_path)
    dvf_array_path = "{0}/temp_dvf.nii".format(output_path)

    for i in range(len(dvf_array)):
        static_image.write(static_image_path)
        dvf_array[i].write(dvf_array_path)

        temp_static = reg.NiftiImageData(static_image_path)
        temp_dvf = reg.NiftiImageData3DDeformation(dvf_array_path)

        resampler = reg.NiftyResample()

        resampler.set_reference_image(temp_static)
        resampler.set_floating_image(temp_static)
        resampler.add_transformation(temp_dvf)

        resampler.set_interpolation_type_to_linear()

        warp = warp_image_forward(resampler, temp_static)

        warped_image = static_image.clone()
        warped_image.fill(warp)

        warped_image.write("{0}/warp_forward_{1}.nii".format(output_path, str(i)))

        difference = temp_static.as_array().astype(np.double) - warp

        difference_image = temp_static.clone()
        difference_image.fill(difference)

        difference_image.write("{0}/warp_forward_difference_{1}.nii".format(output_path, str(i)))

        warp = warp_image_adjoint(resampler, temp_static)

        warped_image = temp_static.clone()
        warped_image.fill(warp)

        warped_image.write("{0}/warp_adjoint_{1}.nii".format(output_path, str(i)))

        difference = temp_static.as_array().astype(np.double) - warp

        difference_image = temp_static.clone()
        difference_image.fill(difference)

        difference_image.write("{0}/warp_adjoint_difference_{1}.nii".format(output_path, str(i)))

    return True


def get_resamplers(static_image, dynamic_array, dvf_array, output_path):
    resamplers = []

    static_image_path = "{0}/temp_static.nii".format(output_path)
    dynamic_array_path = "{0}/temp_dynamic.nii".format(output_path)
    dvf_array_path = "{0}/temp_dvf.nii".format(output_path)

    for j in range(len(dynamic_array)):
        resampler = reg.NiftyResample()

        static_image.write(static_image_path)
        dynamic_array[j].write(dynamic_array_path)
        dvf_array[j].write(dvf_array_path)

        temp_static = reg.NiftiImageData(static_image_path)
        temp_dynamic = reg.NiftiImageData(dynamic_array_path)
        temp_dvf = reg.NiftiImageData3DDeformation(dvf_array_path)

        resampler.set_reference_image(temp_static)
        resampler.set_floating_image(temp_dynamic)
        resampler.add_transformation(temp_dvf)

        resampler.set_interpolation_type_to_linear()

        resamplers.append(resampler)

    return resamplers


def edit_header(data, output_path):
    new_data_array = []

    for i in range(len(data)):
        current_data = nibabel.load(data[i])

        current_data_data = current_data.get_data()
        current_data_affine = current_data.affine
        current_data_header = current_data.header

        current_data_header["intent_code"] = 1007

        new_data = nibabel.Nifti1Image(current_data_data, current_data_affine, current_data_header)

        new_data_path = "{0}/new_dvf_{1}.nii".format(output_path, str(i))

        if os.path.exists(new_data_path):
            os.remove(new_data_path)

        nibabel.save(new_data, new_data_path)

        new_data_array.append(new_data_path)

    return new_data_array


def register_data(static_path, dynamic_path, output_path):
    path_new_displacement_fields = "{0}/new_displacement_fields/".format(output_path)

    if not os.path.exists(path_new_displacement_fields):
        os.makedirs(path_new_displacement_fields, mode=0o770)

    path_new_deformation_fields = "{0}/new_deformation_fields/".format(output_path)

    if not os.path.exists(path_new_deformation_fields):
        os.makedirs(path_new_deformation_fields, mode=0o770)

    path_new_tm = "{0}/new_tm/".format(output_path)

    if not os.path.exists(path_new_tm):
        os.makedirs(path_new_tm, mode=0o770)

    algo = reg.NiftyAladinSym()

    dvf_path = []

    for i in range(len(dynamic_path)):
        ref = eng_ref.ImageData(static_path)
        flo = eng_flo.ImageData(dynamic_path[i])

        algo.set_reference_image(ref)
        algo.set_floating_image(flo)

        algo.process()

        displacement_field = algo.get_displacement_field_forward()
        displacement_field.write("{0}/new_displacement_field_{1}.nii".format(path_new_displacement_fields, str(i)))

        dvf_path.append("{0}/new_DVF_field_{1}.nii".format(path_new_deformation_fields, str(i)))

        deformation_field = algo.get_deformation_field_forward()
        deformation_field.write(dvf_path[i])

        tm = algo.get_transformation_matrix_forward()
        tm.write("{0}/new_tm_{1}.nii".format(path_new_tm, str(i)))

    return dvf_path


def op_test(static_image, output_path):
    static_image_path = "{0}/temp_static.nii".format(output_path)

    static_image.write(static_image_path)

    temp_static = reg.NiftiImageData(static_image_path)

    temp_at = reg.AffineTransformation()

    temp_at_array = temp_at.as_array()
    temp_at_array[0][0] = 1.25
    temp_at_array[1][1] = 1.25
    temp_at_array[2][2] = 1.25
    temp_at_array[3][3] = 1.25

    temp_at = reg.AffineTransformation(temp_at_array)

    resampler = reg.NiftyResample()

    resampler.set_reference_image(temp_static)
    resampler.set_floating_image(temp_static)
    resampler.add_transformation(temp_at)

    resampler.set_interpolation_type_to_linear()

    warp = warp_image_forward(resampler, temp_static)

    warped_image = static_image.clone()
    warped_image.fill(warp)

    warped_image.write("{0}/op_test_warp_forward.nii".format(output_path))

    difference = temp_static.as_array().astype(np.double) - warp

    difference_image = temp_static.clone()
    difference_image.fill(difference)

    difference_image.write("{0}/op_test_warp_forward_difference.nii".format(output_path))

    warp = warp_image_adjoint(resampler, temp_static)

    warped_image = temp_static.clone()
    warped_image.fill(warp)

    warped_image.write("{0}/op_test_warp_adjoint.nii".format(output_path))

    difference = temp_static.as_array().astype(np.double) - warp

    difference_image = temp_static.clone()
    difference_image.fill(difference)

    difference_image.write("{0}/warp_adjoint_difference.nii".format(output_path))

    return True


def get_dvf_path(input_dvf_path, dvf_split):
    all_dvf_path = os.listdir(input_dvf_path)
    dvf_path = []

    for i in range(len(all_dvf_path)):
        current_dvf_path = all_dvf_path[i].rstrip()

        if len(current_dvf_path.split(".nii")) > 1 and len(current_dvf_path.split(dvf_split)) > 1:
            dvf_path.append("{0}/{1}".format(input_dvf_path, current_dvf_path))

    dvf_path.sort(key=human_sorting)

    return dvf_path


def get_data_path(input_dynamic_path, dynamic_split):
    all_dynamic_path = os.listdir(input_dynamic_path)
    dynamic_path = []

    for i in range(len(all_dynamic_path)):
        current_dynamic_path = all_dynamic_path[i].rstrip()

        if len(current_dynamic_path.split(".nii")) > 1 and len(current_dynamic_path.split(dynamic_split)) > 1:
            dynamic_path.append("{0}/{1}".format(input_dynamic_path, current_dynamic_path))

    dynamic_path.sort(key=human_sorting)

    return dynamic_path


def main():
    # file paths to data
    input_data_path = parser.parser(sys.argv[1], "data_path:=")
    data_split = parser.parser(sys.argv[1], "data_split:=")
    input_dvf_path = parser.parser(sys.argv[1], "dvf_path:=")
    dvf_split = parser.parser(sys.argv[1], "dvf_split:=")
    output_path = parser.parser(sys.argv[1], "output_path:=")
    do_op_test = parser.parser(sys.argv[1], "do_op_test:=")
    do_reg = parser.parser(sys.argv[1], "do_reg:=")
    do_test_for_adj = parser.parser(sys.argv[1], "do_test_for_adj:=")

    for i in range(len(input_data_path)):
        if not os.path.exists(output_path[i]):
            os.makedirs(output_path[i], mode=0o770)

        new_dvf_path = "{0}/new_dvfs/".format(output_path[i])

        if not os.path.exists(new_dvf_path):
            os.makedirs(new_dvf_path, mode=0o770)

        # get static and dynamic paths
        dynamic_path = get_data_path(input_data_path[i], data_split[i])

        # load dynamic objects
        dynamic_array = []

        for j in range(len(dynamic_path)):
            dynamic_array.append(reg.NiftiImageData(dynamic_path[j]))

        static_path = "{0}/static_path.nii".format(output_path[i])

        # load static objects
        static_image = reg.NiftiImageData(dynamic_path[0])

        for j in range(1, len(dynamic_path)):
            static_image.fill(static_image.as_array().astype(np.double) + dynamic_array[j].as_array().astype(np.double))

        static_image.write(static_path)

        if bool(distutils.util.strtobool(do_op_test[i])):
            op_test(static_image, output_path[i])

        # if do reg the calc dvf if not load
        if bool(distutils.util.strtobool(do_reg[i])):
            dvf_path = register_data(static_path, dynamic_path, output_path[i])
        else:
            dvf_path = get_dvf_path(input_dvf_path[i], dvf_split[i])

        # fix dvf header and load dvf objects
        dvf_path = edit_header(dvf_path, new_dvf_path)

        dvf_array = []

        for j in range(len(dvf_path)):
            dvf_array.append(reg.NiftiImageData3DDeformation(dvf_path[j]))

        # create object to get forward and adj
        resamplers = get_resamplers(static_image, dynamic_array, dvf_array, output_path[i])

        # test for adj
        if bool(distutils.util.strtobool(do_test_for_adj[i])):
            test_for_adj(static_image, dvf_array, output_path[i])
            output_input(static_image, dynamic_array, dvf_array, output_path[i])

        # initial static image
        initial_static_image = static_image.clone()

        # array to optimise
        optimise_array = static_image.as_array().astype(np.double)

        # array bounds
        bounds = []

        for j in range(len(np.ravel(optimise_array.copy()))):
            bounds.append((-np.inf, np.inf))

        # optimise
        optimise_array = np.reshape(
            scipy.optimize.minimize(objective_function, np.ravel(optimise_array).astype(np.double),
                                    args=(resamplers, dynamic_array, static_image, output_path[i]), method="L-BFGS-B",
                                    jac=gradient_function, bounds=bounds, tol=0.00000000001,
                                    options={"disp": True}).x, optimise_array.shape)

        # output
        static_image.fill(optimise_array)
        static_image.write("{0}/optimiser_output_{1}.nii".format(output_path[i], str(i)))

        difference = static_image.as_array().astype(np.double) - initial_static_image.as_array().astype(np.double)

        difference_image = initial_static_image.clone()
        difference_image.fill(difference)

        static_image.write("{0}/optimiser_output_difference_{1}.nii".format(output_path[i], str(i)))


main()
