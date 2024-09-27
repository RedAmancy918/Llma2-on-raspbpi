** 用于加载模型并与用户交互的示例程序。下一步应该将树莓派上的llama.h传到电脑上进行用例的解析，再重新编译这个代码。**
** is change by llama.h **

#include "llama.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    // 加载模型
    llama_model_params model_params = llama_model_default_params();
    struct llama_model* model = llama_load_model_from_file("/home/team2/llama.cpp/models/7B/llama-2-7b-chat.Q4_K_M.gguf", model_params);
    
    if (!model) {
        std::cerr << "模型加载失败！" << std::endl;
        return 1;
    }

    // 创建上下文
    llama_context_params ctx_params = llama_context_default_params();
    struct llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    
    if (!ctx) {
        std::cerr << "上下文初始化失败！" << std::endl;
        llama_free_model(model);
        return 1;
    }

    // 输入提示
    std::string input_text;

    while (true) {
        // 从用户获取输入
        std::cout << "请输入你的问题（输入'exit'退出）：";
        std::getline(std::cin, input_text);

        // 检查是否退出
        if (input_text == "exit") {
            break;
        }

        // 将输入文本转为 tokens
        std::vector<llama_token> tokens(256);  // 预先分配空间
        int n_tokens = llama_tokenize(model, input_text.c_str(), input_text.size(), tokens.data(), tokens.size(), true, false);
        
        // 检查 token 的数量
        if (n_tokens < 0) {
            std::cerr << "token 化失败！" << std::endl;
            continue;
        }

        // 调整 tokens 向量大小
        tokens.resize(n_tokens);

        // 推理
        llama_eval(ctx, tokens.data(), tokens.size(), 1);

        // 生成输出
        std::string output_text;
        llama_token token;
        while ((token = llama_sample_token(ctx)) != llama_token_eos(model)) {
            output_text += llama_token_to_str(model, token);
        }

        // 输出模型的响应
        std::cout << "模型响应: " << output_text << std::endl;
    }

    // 释放资源
    llama_free(ctx);
    llama_free_model(model);
    return 0;
}
