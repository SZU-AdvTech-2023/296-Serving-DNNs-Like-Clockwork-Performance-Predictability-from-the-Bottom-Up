#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>
#include <sstream>

#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/model/batched.h"
#include <catch2/catch.hpp>

using namespace clockwork::model;

char* batch(std::string data, int batch_size, int batch_size_with_padding) {
    char* batched_data = static_cast<char*>(malloc(data.size() * batch_size_with_padding));
    for (unsigned i = 0; i < batch_size; i++) {
        std::memcpy(batched_data + data.size() * i, data.data(), data.size());
    }

    // Check
    for (unsigned i = 0; i < batch_size; i++) {
        for (unsigned j = 0; j < data.size(); j++) {
            REQUIRE(batched_data[i * data.size() + j] == data[j]);
        }
    }

    return batched_data;
}


TEST_CASE("Batched models individually", "[batched]") {
    // This test currently fails because the batched versions of the models are outputting incorrect answers.
    // Warrants further investigation
    for (unsigned batch_size = 1; batch_size <= 16; batch_size *= 2) {
        INFO("Checking batch_size=" << batch_size);

        int weights_page_size = 16 * 1024 * 1024;
        int workspace_page_size = 64 * 1024 * 1024;
        int input_size = 224*224*3*4;
        int output_size = 1000 * 1 * 4;

        std::string f = clockwork::util::get_example_batched_model();
        std::stringstream batched_f;
        batched_f << f << "." << batch_size;

        Model* model = Model::loadFromDisk(batched_f.str()+".so", batched_f.str()+".clockwork", f+".clockwork_params", GPU_ID_0);

        model->instantiate_model_on_host();
        model->instantiate_model_on_device();

        auto weights_pages = make_cuda_pages(weights_page_size, model->num_weights_pages(weights_page_size));
        model->transfer_weights_to_device(weights_pages->pages, NULL);
        cuda_synchronize(NULL);

        std::string single_input, single_output;
        clockwork::util::readFileAsString(f+".input", single_input);
        clockwork::util::readFileAsString(f+".output", single_output);

        REQUIRE(single_input.size() == input_size);
        REQUIRE(single_output.size() == output_size);
        REQUIRE(model->input_size() == input_size * batch_size);
        REQUIRE(model->output_size() == output_size * batch_size);

        char* input = batch(single_input, batch_size, batch_size);
        char* expected_output = batch(single_output, batch_size, batch_size);

        char actual_output[output_size * batch_size];

        auto io_alloc = make_cuda_pages(model->io_memory_size(), 1);
        auto workspace_alloc = make_cuda_pages(model->workspace_memory_size(), 1);

        model->transfer_input_to_device(input, io_alloc->ptr, NULL);
        cuda_synchronize(NULL);
        model->call(weights_pages->pages, io_alloc->ptr, workspace_alloc->ptr, NULL);
        cuda_synchronize(NULL);
        model->transfer_output_from_device(actual_output, io_alloc->ptr, NULL);
        cuda_synchronize(NULL);

        float* actualOutputF = static_cast<float*>(static_cast<void*>(actual_output));
        float* expectedOutputF = static_cast<float*>(static_cast<void*>(expected_output));


        for (unsigned i = 0; i < batch_size; i++) {
            int offset = i * 1000;
            auto max_index_actual = std::distance(actualOutputF + offset, std::max_element(actualOutputF + offset, actualOutputF + offset + 1000));
            auto max_index_expect = std::distance(expectedOutputF + offset, std::max_element(expectedOutputF + offset, expectedOutputF + offset + 1000));
            INFO("batch_size=" << batch_size << " i=" << i);
            REQUIRE(max_index_expect == max_index_actual);
        }

        for (unsigned i = 0; i < (output_size * batch_size)/4; i++) {
            INFO("batch_size=" << batch_size << " i=" << i);
            REQUIRE(actualOutputF[i] == expectedOutputF[i]);
        }

        delete model;
        free(input);
        free(expected_output);
    }
}


TEST_CASE("Batched models with partial batches", "[batched]") {
    // This test currently fails because the batched versions of the models are outputting incorrect answers.
    // Warrants further investigation

    int weights_page_size = 16 * 1024 * 1024;
    int workspace_page_size = 64 * 1024 * 1024;
    int input_size = 224*224*3*4;
    int output_size = 1000 * 1 * 4;

    std::string f = clockwork::util::get_example_batched_model();

    BatchedModel* model = BatchedModel::loadFromDisk(f, GPU_ID_0);
    
    model->instantiate_models_on_host();
    model->instantiate_models_on_device();

    // Check expected batch sizes
    std::vector<unsigned> batch_sizes = model->implemented_batch_sizes();
    REQUIRE(batch_sizes.size() == 5);
    unsigned i = 0;
    unsigned batch_size = 1;
    for (; i < batch_sizes.size(); i++) {
        REQUIRE(batch_sizes[i] == batch_size);
        batch_size *= 2;
    }
    REQUIRE(model->max_batch_size() == 16);


    int num_weights_pages = model->num_weights_pages(weights_page_size);
    auto weights_pages = make_cuda_pages(weights_page_size, num_weights_pages);
    model->transfer_weights_to_device(weights_pages->pages, NULL);

    std::string single_input, single_output;
    clockwork::util::readFileAsString(f+".input", single_input);
    clockwork::util::readFileAsString(f+".output", single_output);
    REQUIRE(single_input.size() == input_size);
    REQUIRE(single_output.size() == output_size);

    for (unsigned batch_size = 1; batch_size <= batch_sizes[batch_sizes.size()-1]; batch_size++) {

        REQUIRE(model->input_size(batch_size) == input_size * batch_size);
        REQUIRE(model->output_size(batch_size) == output_size * batch_size);

        char* input = batch(single_input, batch_size, model->padded_batch_size(batch_size));
        char* expected_output = batch(single_output, batch_size, model->padded_batch_size(batch_size));
        char actual_output[output_size * model->padded_batch_size(batch_size)];

        auto io_alloc = make_cuda_pages(model->io_memory_size(batch_size), 1);
        auto workspace_alloc = make_cuda_pages(model->workspace_memory_size(batch_size), 1);

        model->transfer_input_to_device(batch_size, input, io_alloc->ptr, NULL);
        cuda_synchronize(NULL);
        model->call(batch_size, weights_pages->pages, io_alloc->ptr, workspace_alloc->ptr, NULL);
        cuda_synchronize(NULL);
        model->transfer_output_from_device(batch_size, actual_output, io_alloc->ptr, NULL);
        cuda_synchronize(NULL);

        float* actualOutputF = static_cast<float*>(static_cast<void*>(actual_output));
        float* expectedOutputF = static_cast<float*>(static_cast<void*>(expected_output));


        for (unsigned i = 0; i < batch_size; i++) {
            int offset = i * 1000;
            auto max_index_actual = std::distance(actualOutputF + offset, std::max_element(actualOutputF + offset, actualOutputF + offset + 1000));
            auto max_index_expect = std::distance(expectedOutputF + offset, std::max_element(expectedOutputF + offset, expectedOutputF + offset + 1000));
            INFO("batch_size=" << batch_size << " i=" << i);
            REQUIRE(max_index_expect == max_index_actual);
        }

        for (unsigned i = 0; i < (output_size * batch_size)/4; i++) {
            INFO("batch_size=" << batch_size << " i=" << i);
            REQUIRE(actualOutputF[i] == expectedOutputF[i]);
        }
        free(input);
        free(expected_output);
    }

    delete model;
}