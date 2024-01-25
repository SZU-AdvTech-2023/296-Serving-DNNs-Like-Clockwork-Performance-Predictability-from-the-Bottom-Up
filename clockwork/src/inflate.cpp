#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include "clockwork/telemetry.h"
#include "clockwork/common.h"
#include <fstream>
#include <sstream>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>

/** Inflates output clockwork telemetry to TSV file */
void inflate_task(std::string input_filename, std::string output_filename)
{
    std::ifstream infile;
    infile.open(input_filename);

    pods::InputStream in(infile);
    pods::BinaryDeserializer<decltype(in)> deserializer(in);

    uint64_t start_time = 0;

    clockwork::SerializedTaskTelemetry t;
    int count = 0;

    std::vector<std::string> headers = {{"action_id",
                                         "action_type",
										 "task_type",
										 "executor_id",
										 "gpu_id",
										 "status",
                                         "model_id",
										 "batch_size",
                                         "enqueued",
                                         "eligible_for_dequeue",
                                         "dequeued",
                                         "exec_complete",
                                         "async_complete",
                                         "async_wait",
                                         "async_duration",
                                         "queue_latency",
                                         "total_latency"}};

    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (deserializer.load(t) == pods::Error::NoError)
    {
        std::unordered_map<std::string, std::string> row;

		row["action_type"] = std::to_string(t.action_type);
		row["task_type"] = std::to_string(t.task_type);
        row["action_id"] = std::to_string(t.action_id);
        row["status"] = std::to_string(t.status);
        row["executor_id"] = std::to_string(t.executor_id);
        row["gpu_id"] = std::to_string(t.gpu_id);
        row["model_id"] = std::to_string(t.model_id);
        row["batch_size"] = std::to_string(t.batch_size);
        row["enqueued"] = std::to_string(t.enqueued);
        row["eligible_for_dequeue"] = std::to_string(t.eligible_for_dequeue);
        row["dequeued"] = std::to_string(t.dequeued);
        row["exec_complete"] = std::to_string(t.exec_complete);
        row["async_complete"] = std::to_string(t.async_complete);
        row["async_wait"] = std::to_string(t.async_wait);
        row["async_duration"] = std::to_string(t.async_duration);
        row["queue_latency"] = std::to_string(t.dequeued - t.enqueued);
        row["total_latency"] = std::to_string(t.async_complete - t.enqueued);

        rows.push_back(row);
    }
    std::cout << "Processed " << rows.size() << " records" << std::endl;

    std::ofstream outfile;
    outfile.open(output_filename);

    int i = 0;
    for (auto header : headers)
    {
        if (i++ > 0)
        {
            outfile << "\t";
        }
        outfile << header;
    }
    outfile << "\n";

    for (auto row : rows)
    {
        i = 0;
        for (auto header : headers)
        {
            if (i++ > 0)
                outfile << "\t";
            if (row.find(header) != row.end())
                outfile << row[header];
        }
        outfile << "\n";
    }

    outfile.close();
}

void inflate_action(std::string input_filename, std::string output_filename)
{
    std::ifstream infile;
    infile.open(input_filename);

    pods::InputStream in(infile);
    pods::BinaryDeserializer<decltype(in)> deserializer(in);

    uint64_t start_time = 0;

    clockwork::SerializedActionTelemetry t;
    int count = 0;

    std::vector<std::string> headers = {{"telemetry_type",
										"action_id",
										 "action_type",
										 "status",
										 "timestamp"}};


    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (deserializer.load(t) == pods::Error::NoError)
    {
        std::unordered_map<std::string, std::string> row;

		row["telemetry_type"] = std::to_string(t.telemetry_type);
		row["action_id"] = std::to_string(t.action_id);
		row["action_type"] = std::to_string(t.action_type);
		row["status"] = std::to_string(t.status);
		row["timestamp"] = std::to_string(t.timestamp);

		rows.push_back(row);
    }

    std::cout << "Processed " << rows.size() << " records" << std::endl;

    std::ofstream outfile;
    outfile.open(output_filename);

    int i = 0;
    for (auto header : headers)
    {
        if (i++ > 0)
        {
            outfile << "\t";
        }
        outfile << header;
    }
    outfile << "\n";

    for (auto row : rows)
    {
        i = 0;
        for (auto header : headers)
        {
            if (i++ > 0)
                outfile << "\t";
            if (row.find(header) != row.end())
                outfile << row[header];
        }
        outfile << "\n";
    }

    outfile.close();
}

void inflate_request(std::string input_filename, std::string output_filename)
{
    std::ifstream infile;
    infile.open(input_filename);

    pods::InputStream in(infile);
    pods::BinaryDeserializer<decltype(in)> deserializer(in);

    uint64_t start_time = 0;

    clockwork::SerializedRequestTelemetry t;
    int count = 0;

    std::vector<std::string> headers = {{
        "request_id",
        "model_id",
        "arrived",
        "submitted",
        "complete",
        "execution_latency",
        "total_latency",
    }};

    for (unsigned i = 0; i < clockwork::TaskTypes.size(); i++)
    {
        std::string task_type = clockwork::TaskTypeName(clockwork::TaskTypes[i]);
        headers.push_back(task_type);
        headers.push_back(task_type + "_queue");
        headers.push_back(task_type + "_sync");
        headers.push_back(task_type + "_async");
    }

    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (deserializer.load(t) == pods::Error::NoError)
    {
        std::unordered_map<std::string, std::string> row;

        row["request_id"] = std::to_string(t.request_id);
        row["model_id"] = std::to_string(t.model_id);
        row["arrived"] = std::to_string(t.arrived);
        row["submitted"] = std::to_string(t.submitted);
        row["complete"] = std::to_string(t.complete);
        row["execution_latency"] = std::to_string(t.complete - t.submitted);
        row["total_latency"] = std::to_string(t.complete - t.arrived);

        for (unsigned i = 0; i < t.tasks.size(); i++)
        {
            clockwork::SerializedTaskTelemetry task = t.tasks[i];
            std::string task_type = clockwork::TaskTypeName(clockwork::TaskTypes[task.task_type]);

            if (i < t.tasks.size() - 1)
            {
                row[task_type] = std::to_string(t.tasks[i + 1].enqueued - task.dequeued);
            }
            else
            {
                row[task_type] = std::to_string(t.complete - task.dequeued);
            }
            row[task_type + "_queue"] = std::to_string(task.dequeued - task.eligible_for_dequeue);
            row[task_type + "_sync"] = std::to_string(task.exec_complete - task.dequeued);
            row[task_type + "_async"] = std::to_string(task.async_duration);
        }

        rows.push_back(row);
    }
    std::cout << "Processed " << rows.size() << " records" << std::endl;

    std::ofstream outfile;
    outfile.open(output_filename);

    int i = 0;
    for (auto header : headers)
    {
        if (i++ > 0)
        {
            outfile << "\t";
        }
        outfile << header;
    }
    outfile << "\n";

    for (auto row : rows)
    {
        i = 0;
        for (auto header : headers)
        {
            if (i++ > 0)
                outfile << "\t";
            if (row.find(header) != row.end())
                outfile << row[header];
        }
        outfile << "\n";
    }

    outfile.close();
}

void show_usage()
{
    std::cout << "Inflates a binary format telemetry file into a TSV" << std::endl;
    std::cout << "./inflate [input_file] [outputfile] [request/task/action]" << std::endl;
}

int main(int argc, char *argv[])
{
    std::vector<std::string> non_argument_strings;

    if (argc < 4)
    {
        show_usage();
        return 0;
    }

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            show_usage();
            return 0;
        }
        else
        {
            non_argument_strings.push_back(arg);
        }
    }

    if (non_argument_strings.size() < 1)
    {
        std::cerr << "Expected input telemetry filename, none given." << std::endl
                  << "Execute with --help for usage information." << std::endl;
        return 1;
    }

    std::string input_filename = non_argument_strings[0];
    std::string output_filename = input_filename + ".tsv";
    if (non_argument_strings.size() >= 2)
    {
        output_filename = non_argument_strings[1];
    }

    std::cout << "Inflating " << input_filename << std::endl
              << "       to " << output_filename << std::endl;

    if (non_argument_strings[2] == std::string("request"))
    {
        inflate_request(input_filename, output_filename);
    }
    else if (non_argument_strings[2] == std::string("task"))
    {
        inflate_task(input_filename, output_filename);
    }
	else if (non_argument_strings[2] == std::string("action"))
	{
		inflate_action(input_filename, output_filename);
	}

    return 0;
}
