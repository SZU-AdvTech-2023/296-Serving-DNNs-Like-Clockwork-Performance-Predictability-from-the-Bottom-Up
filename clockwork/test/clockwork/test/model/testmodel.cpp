#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/test/actions.h"
#include <catch2/catch.hpp>

using namespace clockwork::model;

void assert_is_cool(Model* model) {
    size_t weights_page_size = 16 * 1024 * 1024;
    size_t workspace_page_size = 64 * 1024 * 1024;
    size_t input_size = 224*224*3*4;
    size_t output_size = 1000 * 1 * 4;
    char input[input_size];
    char output[output_size];

    std::vector<char*> weights_pages;
    char* io_memory = nullptr;
    char* workspace_memory = nullptr;


    REQUIRE_THROWS(model->uninstantiate_model_on_host());
    REQUIRE_THROWS(model->num_weights_pages(weights_page_size));
    REQUIRE_THROWS(model->workspace_memory_size());
    REQUIRE_THROWS(model->io_memory_size());
    REQUIRE_THROWS(model->input_size());
    REQUIRE_THROWS(model->output_size());
    REQUIRE_THROWS(model->call(weights_pages, io_memory, workspace_memory, NULL));
}

void assert_is_warm(Model* model) {
    size_t weights_page_size = 16 * 1024 * 1024;
    size_t workspace_page_size = 64 * 1024 * 1024;
    size_t input_size = 224*224*3*4;
    size_t output_size = 1000 * 1 * 4;
    char input[input_size];
    char output[output_size];

    int num_weights_pages = 5;
    size_t io_memory_size = 606112;
    size_t workspace_memory_size = 8531968;

    std::vector<char*> weights_pages;
    char* io_memory = nullptr;
    char* workspace_memory = nullptr;

    REQUIRE_THROWS(model->instantiate_model_on_host());
    REQUIRE_THROWS(model->call(weights_pages, io_memory, workspace_memory, NULL));

    REQUIRE(num_weights_pages == model->num_weights_pages(weights_page_size));
    REQUIRE(io_memory_size == model->io_memory_size());
    REQUIRE(workspace_memory_size == model->workspace_memory_size());
    REQUIRE(input_size == model->input_size());
    REQUIRE(output_size == model->output_size());
}

void assert_is_hot(Model* model) {
    size_t weights_page_size = 16 * 1024 * 1024;
    size_t workspace_page_size = 64 * 1024 * 1024;
    size_t input_size = 224*224*3*4;
    size_t output_size = 1000 * 1 * 4;
    char input[input_size];
    char output[output_size];

    int num_weights_pages = 5;
    size_t io_memory_size = 606112;
    size_t workspace_memory_size = 8531968;

    auto weights_pages = make_cuda_pages(weights_page_size, num_weights_pages);
    auto io_alloc = make_cuda_pages(io_memory_size, 1);
    auto workspace_alloc = make_cuda_pages(workspace_memory_size, 1);
 
    REQUIRE(num_weights_pages == model->num_weights_pages(weights_page_size));
    REQUIRE(io_memory_size == model->io_memory_size());
    REQUIRE(workspace_memory_size == model->workspace_memory_size());
    REQUIRE(input_size == model->input_size());
    REQUIRE(output_size == model->output_size());

    REQUIRE_NOTHROW(model->transfer_weights_to_device(weights_pages->pages, NULL));
    REQUIRE_NOTHROW(model->transfer_input_to_device(input, io_alloc->ptr, NULL));
    REQUIRE_NOTHROW(model->call(weights_pages->pages, io_alloc->ptr, workspace_alloc->ptr, NULL));
    REQUIRE_NOTHROW(model->transfer_output_from_device(output, io_alloc->ptr, NULL));

    cuda_synchronize(NULL);
}

TEST_CASE("Load model from disk", "[model]") {

    std::string f = clockwork::util::get_example_model();

    Model* model = nullptr;
    REQUIRE_NOTHROW(model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0));
    REQUIRE(model != nullptr);
    delete model;

}

TEST_CASE("Model lifecycle 1", "[model]") {

    std::string f = clockwork::util::get_example_model();

    Model* model = nullptr;
    REQUIRE_NOTHROW(model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0));
    REQUIRE(model != nullptr);

    int page_size;

    REQUIRE_THROWS(model->uninstantiate_model_on_host());

    REQUIRE_THROWS(model->num_weights_pages(page_size));
    REQUIRE_THROWS(model->io_memory_size());
    REQUIRE_THROWS(model->workspace_memory_size());

    REQUIRE_NOTHROW(model->instantiate_model_on_host());
    REQUIRE_NOTHROW(model->uninstantiate_model_on_host());

    // Instantiate on host can only happen once now
    REQUIRE_THROWS(model->instantiate_model_on_host());

    delete model;
}


TEST_CASE("Model Lifecycle 2", "[model] [model_lifecycle_2]") {

    std::string f = clockwork::util::get_example_model();

    Model* model = nullptr;
    REQUIRE_NOTHROW(model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0));
    REQUIRE(model != nullptr);
    
    assert_is_cool(model);

    model->instantiate_model_on_host();

    assert_is_warm(model);

    model->instantiate_model_on_device();

    assert_is_hot(model);

    model->uninstantiate_model_on_device();

    assert_is_warm(model);

    model->instantiate_model_on_device();

    assert_is_hot(model);

    model->uninstantiate_model_on_device();

    assert_is_warm(model);

    model->uninstantiate_model_on_host();

    assert_is_cool(model);

    /*
    We no longer support multiple instantiations on host.  Just once.
    */

    REQUIRE_THROWS(model->instantiate_model_on_host());

    delete model;

}

TEST_CASE("Model produces correct output", "[e2e] [model]") {

    int weights_page_size = 16 * 1024 * 1024;
    int workspace_page_size = 64 * 1024 * 1024;
    int input_size = 224*224*3*4;
    int output_size = 1000 * 1 * 4;

    int num_weights_pages = 5;
    size_t io_memory_size = 606112;
    size_t workspace_memory_size = 8531968;

    auto weights_pages = make_cuda_pages(weights_page_size, num_weights_pages);
    auto io_alloc = make_cuda_pages(io_memory_size, 1);
    auto workspace_alloc = make_cuda_pages(workspace_memory_size, 1);

    std::string f = clockwork::util::get_example_model();

    Model* model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0);
    
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    model->transfer_weights_to_device(weights_pages->pages, NULL);

    std::ifstream in(f+".input");
    std::string input_filename = f+".input";
    std::string output_filename = f+".output";
    std::string input, expectedOutput;
    char actualOutput[output_size];
    clockwork::util::readFileAsString(input_filename, input);
    clockwork::util::readFileAsString(output_filename, expectedOutput);

    REQUIRE(input.size() == input_size);
    model->transfer_input_to_device(input.data(), io_alloc->ptr, NULL);
    model->call(weights_pages->pages, io_alloc->ptr, workspace_alloc->ptr, NULL);
    model->transfer_output_from_device(actualOutput, io_alloc->ptr, NULL);

    REQUIRE(output_size == expectedOutput.size());

    cuda_synchronize(NULL);

    float* actualOutputF = static_cast<float*>(static_cast<void*>(actualOutput));
    const float* expectedOutputF = reinterpret_cast<const float*>(expectedOutput.data());

    auto max_index_actual = std::distance(actualOutputF, std::max_element(actualOutputF, actualOutputF + 1000));
    auto max_index_expect = std::distance(expectedOutputF, std::max_element(expectedOutputF, expectedOutputF + 1000));
    REQUIRE(max_index_expect == max_index_actual);

    for (unsigned i = 0; i < output_size/4; i++) {
        REQUIRE(actualOutputF[i] == expectedOutputF[i]);
    }
}

TEST_CASE("Batched model produces correct output", "[e2e2] [model]") {

    int weights_page_size = 16 * 1024 * 1024;
    int workspace_page_size = 64 * 1024 * 1024;
    int input_size = 2*224*224*3*4;
    int output_size = 1000 * 2 * 4;

    std::string f = clockwork::util::get_example_batched_model();

    Model* model = Model::loadFromDisk(f+".2.so", f+".2.clockwork", f+".clockwork_params", GPU_ID_0);
    
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();

    auto weights_pages = make_cuda_pages(weights_page_size, model->num_weights_pages(weights_page_size));
    model->transfer_weights_to_device(weights_pages->pages, NULL);



    std::string input, expectedOutput;
    clockwork::util::readFileAsString(f+".input", input);
    clockwork::util::readFileAsString(f+".output", expectedOutput);

    REQUIRE(input_size == 2 * input.size());
    REQUIRE(output_size == 2 * expectedOutput.size());

    char batched_input[input_size];
    std::memcpy(batched_input, input.data(), input.size());
    std::memcpy(batched_input + input.size(), input.data(), input.size());

    char batched_expected_output[output_size];
    std::memcpy(batched_expected_output, expectedOutput.data(), expectedOutput.size());
    std::memcpy(batched_expected_output + expectedOutput.size(), expectedOutput.data(), expectedOutput.size());

    auto io_alloc = make_cuda_pages(model->io_memory_size(), 1);
    auto workspace_alloc = make_cuda_pages(model->workspace_memory_size(), 1);

    model->transfer_input_to_device(batched_input, io_alloc->ptr, NULL);
    model->call(weights_pages->pages, io_alloc->ptr, workspace_alloc->ptr, NULL);

    char actualOutput[output_size];
    model->transfer_output_from_device(actualOutput, io_alloc->ptr, NULL);

    cuda_synchronize(NULL);

    float* actualOutputF = static_cast<float*>(static_cast<void*>(actualOutput));
    float* expectedOutputF = static_cast<float*>(static_cast<void*>(batched_expected_output));

    auto max_index_actual_1 = std::distance(actualOutputF, std::max_element(actualOutputF, actualOutputF + 1000));
    auto max_index_expect_1 = std::distance(expectedOutputF, std::max_element(expectedOutputF, expectedOutputF + 1000));
    REQUIRE(max_index_expect_1 == max_index_actual_1);

    auto max_index_actual_2 = std::distance(actualOutputF+1000, std::max_element(actualOutputF+1000, actualOutputF + 2000));
    auto max_index_expect_2 = std::distance(expectedOutputF+1000, std::max_element(expectedOutputF+1000, expectedOutputF + 2000));
    REQUIRE(max_index_expect_2 == max_index_actual_2);

    for (unsigned i = 0; i < output_size/4; i++) {
        REQUIRE(actualOutputF[i] == expectedOutputF[i]);
    }
}

TEST_CASE("Load multiple models", "[model]") {

	auto clockwork = std::make_shared<ClockworkRuntimeWrapper>();
    std::string path = clockwork::util::get_example_batched_model();
	unsigned gpu_id = 0;
	int copies = 5;
	std::vector<BatchedModel*> batched_models = BatchedModel::loadMultipleFromDisk(path, gpu_id, copies);

	assert(batched_models.size() == copies);
	for (auto batched_model : batched_models){
		batched_model->instantiate_models_on_host();
		batched_model->instantiate_models_on_device();
	}

	for (int i = 0; i < copies; i ++) {
		RuntimeModel* rm = new RuntimeModel(batched_models[i], gpu_id);
		if (!clockwork->manager->models->put_if_absent(i, gpu_id, rm)) {
			throw TaskError(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
		}
	}

	int i = 0;
	for (auto model : batched_models) {
		TestLoadWeightsAction load_weights(clockwork.get(), load_weights_action(i));
		load_weights.submit();
		load_weights.await();
		load_weights.check_success(true);

		TestInferAction infer(clockwork.get(), infer_action(1, model));
		infer.submit();
		infer.await();
		infer.check_success(true);
		i++;
	}

}


