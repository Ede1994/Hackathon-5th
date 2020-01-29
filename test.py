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

import parser


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


def objective_function(optimise_array, resampler, dynamic_images, static_image):
    static_image.fill(np.reshape(optimise_array, static_image.as_array().shape))

    objective_value = 0.0

    for i in range(len(dynamic_images)):
        objective_value = objective_value + (
                    np.nansum(np.square(dynamic_images[i].as_array() - warp_image_forward(resampler[i], static_image)),
                              dtype=np.double) / 2.0)

    print("Objective function value: {0}".format(str(objective_value)))

    return objective_value


def gradient_function(optimise_array, resampler, dynamic_images, static_image, output_path):
    static_image.fill(np.reshape(optimise_array, static_image.as_array().shape))

    gradient_value = static_image.clone()
    gradient_value.fill(0.0)

    adjoint_image = static_image.clone()

    for i in range(len(dynamic_images)):
        static_image.write("{0}/temp_static.nii".format(output_path))
        dynamic_images[i].write("{0}/temp_dynamic.nii".format(output_path))

        temp_static = reg.NiftiImageData("{0}/temp_static.nii".format(output_path))
        temp_dynamic = reg.NiftiImageData("{0}/temp_dynamic.nii".format(output_path))

        adjoint_image.fill(warp_image_forward(resampler[i], temp_static) - temp_dynamic.as_array())

        gradient_value.fill(gradient_value.as_array() + warp_image_adjoint(resampler[i], adjoint_image))

    gradient_value.write("{0}/gradient.nii".format(output_path))

    print("Max gradient value: {0}, Min gradient value: {1}, Mean gradient value: {2}, Gradient norm: {3}".format(
        str(gradient_value.as_array().max()), str(gradient_value.as_array().min()),
        str(np.nanmean(gradient_value.as_array())), str(np.linalg.norm(gradient_value.as_array()))))

    return np.ravel(gradient_value.as_array())


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
        current_dynamic_path = dynamic_path[i]

        ref = reg.NiftiImageData(static_path)
        flo = reg.NiftiImageData(current_dynamic_path)

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


def test_for_adj(static_image, dynamic_array, dvf_array, output_path):
    static_image_path = "{0}/temp_static.nii".format(output_path)
    dynamic_array_path = "{0}/temp_dynamic.nii".format(output_path)
    dvf_array_path = "{0}/temp_dvf.nii".format(output_path)
    
    for i in range(len(dynamic_array)):
        static_image.write(static_image_path)
        dynamic_array[i].write(dynamic_array_path)
        dvf_array[i].write(dvf_array_path)

        temp_static = reg.NiftiImageData(static_image_path)
        temp_dynamic = reg.NiftiImageData(dynamic_array_path)
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

        difference = temp_static.as_array() - warp

        difference_image = temp_static.clone()
        difference_image.fill(difference)

        difference_image.write("{0}/warp_forward_difference_{1}.nii".format(output_path, str(i)))

        warp = warp_image_adjoint(resampler, temp_static)

        warped_image = temp_dynamic.clone()
        warped_image.fill(warp)

        warped_image.write("{0}/warp_adjoint_{1}.nii".format(output_path, str(i)))

        difference = temp_static.as_array() - warp

        difference_image = temp_static.clone()
        difference_image.fill(difference)

        difference_image.write("{0}/warp_adjoint_difference_{1}.nii".format(output_path, str(i)))

    return True


def get_data_path(data_path):
    all_dynamic_path = os.listdir(data_path)
    dynamic_path = []

    for i in range(len(all_dynamic_path)):
        current_dynamic_path = all_dynamic_path[i].rstrip()

        if len(current_dynamic_path.split(".nii")) > 1 and len(current_dynamic_path.split("fixed")) > 1:
            dynamic_path.append("{0}/{1}".format(data_path, current_dynamic_path))

    dynamic_path.sort(key=human_sorting)

    return dynamic_path


def get_dvf_path(dvf_path):
    all_dvf_path = os.listdir(dvf_path)
    dvf_path = []

    for i in range(len(all_dvf_path)):
        current_dvf_path = all_dvf_path[i].rstrip()

        if len(current_dvf_path.split("DVF")) > 1:
            dvf_path.append("{0}/{1}".format(dvf_path, current_dvf_path))

    dvf_path.sort(key=human_sorting)

    return dvf_path


def main():
    # file paths to data
    data_path = parser.parser(sys.argv[1], "data_path:=")[0]
    dvf_path = parser.parser(sys.argv[1], "dvf_path:=")[0]
    output_path = parser.parser(sys.argv[1], "output_path:=")[0]
    do_reg = bool(distutils.util.strtobool(parser.parser(sys.argv[1], "do_reg:=")[0]))
    do_test_for_adj = bool(distutils.util.strtobool(parser.parser(sys.argv[1], "do_test_for_adj:=")[0]))

    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o770)


    new_dvf_path = "{0}/new_dvfs/".format(output_path)

    # get static and dynamic paths
    dynamic_path = get_data_path(data_path)

    # load dynamic objects
    dynamic_array = []

    for i in range(len(dynamic_path)):
        dynamic_array.append(reg.NiftiImageData(dynamic_path[i]))

    static_path = "./data/static_path.nii"

    # load static objects
    static_image = reg.NiftiImageData(dynamic_path[0])

    for i in range(1, len(dynamic_path)):
        static_image.fill(static_image.as_array() + dynamic_array[i].as_array())

    static_image.write(static_path)

    # if do reg the calc dvf if not load
    if do_reg:
        dvf_path = register_data(static_path, dynamic_path, output_path)
    else:
        dvf_path = get_dvf_path(dvf_path)

    # fix dvf header and load dvf objects
    dvf_path = edit_header(dvf_path, new_dvf_path)

    dvf_array = []

    for i in range(len(dvf_path)):
        dvf_array.append(reg.NiftiImageData3DDeformation(dvf_path[i]))

    # create object to get forward and adj
    resamplers = []

    for i in range(len(dynamic_array)):
        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(dynamic_array[i])
        resampler.add_transformation(dvf_array[i])

        resampler.set_interpolation_type_to_cubic_spline()

        resamplers.append(resampler)

    # test for adj
    if do_test_for_adj:
        test_for_adj(static_image, dynamic_array, dvf_array, output_path)

    # array to optimise
    optimise_array = static_image.as_array()

    # optimise
    optimise_array = np.reshape(scipy.optimize.minimize(objective_function, np.ravel(optimise_array),
                                                        args=(resamplers, dynamic_array, static_image, output_path),
                                                        method="L-BFGS-B", jac=gradient_function,
                                                        bounds=(-np.inf, np.inf), tol=1.0,
                                                        options={"disp": True, "maxiter": -10, "gtol": 1.0}).x,
                                optimise_array.shape)

    # output
    static_image.fill(optimise_array)
    static_image.write("{0}/optimiser_output.nii")


main()
