#include "parser/tensorflow_parser.h"
#include "graph/ascend_string.h"
#include "external/aoe.h"

using namespace Aoe;

bool ParseModel(const std::string &model, ge::Graph &graph)
{
    std::map<ge::AscendString, ge::AscendString> options;
    auto ret = ge::aclgrphParseTensorFlow(model.c_str(), options, graph);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("[ERROR] Parse model failed, ret = %d.\n", ret);
        return false;
    }
    return true;
}

std::map<ge::AscendString, ge::AscendString> Convert(const std::map<std::string, std::string> &options)
{
    std::map<ge::AscendString, ge::AscendString> result;
    for (const auto &kv : options) {
        result.emplace(kv.first.c_str(), kv.second.c_str());
    }
    return result;
}

int main(int argc, char *args[])
{
    printf("[INFO] Welcome to test aoe APIs.\n");

    const std::map<std::string, std::string> globalOptions = {
        {"job_type", "2"},
        {"framework", "3"},
    };
    AoeStatus status = AoeInitialize(Convert(globalOptions));
    if (status != AOE_SUCCESS) {
        printf("[ERROR] AoeInitialize failed, ret = %d.\n", status);
        return -1;
    }

    uint64_t sessionId = 0;
    status = AoeCreateSession(sessionId);
    if (status != AOE_SUCCESS) {
        printf("[ERROR] AoeCreateSession failed, ret = %d.\n", status);
        AoeFinalize();
        return -1;
    }

    std::string filePath = "./test.pb";
    ge::Graph graph;
    if (!ParseModel(filePath, graph)) {
        printf("[ERROR] Parse model failed.\n");
        AoeDestroySession(sessionId);
        AoeFinalize();
        return -1;
    }
    status = AoeSetTuningGraph(sessionId, graph);
    if (status != AOE_SUCCESS) {
        printf("[ERROR] AoeSetTuningGraph failed, ret = %d.\n", status);
        AoeDestroySession(sessionId);
        AoeFinalize();
        return -1;
    }

    const std::map<std::string, std::string> tuningOptions = {};
    status = AoeTuningGraph(sessionId, Convert(tuningOptions));
    if (status != AOE_SUCCESS && status != AOE_ERROR_DYNAMIC_GRAPH && status != AOE_ERROR_DYNAMIC_SHAPE_RANGE) {
        printf("[ERROR] AoeTuningGraph failed, ret = %d.\n", status);
        AoeDestroySession(sessionId);
        AoeFinalize();
        return -1;
    }

    status = AoeDestroySession(sessionId);
    if (status != AOE_SUCCESS) {
        printf("[ERROR] AoeDestroySession failed, ret = %d.\n", status);
        AoeFinalize();
        return -1;
    }

    status = AoeFinalize();
    if (status != AOE_SUCCESS) {
        printf("[ERROR] AoeFinalize failed, ret = %d.\n", status);
        return -1;
    }

    printf("[INFO] Test aoe APIs successfully.\n");
    return 0;
}