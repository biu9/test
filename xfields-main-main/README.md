在原来文件基础上修改：

- results中为训练好的模型，建议不要更改
- 使用test.py将模型转为SaveModel格式
- 使用converter.py将SaveModel转为tflite格式，文件在results/juice下
- lite_test和原来的test功能相同，用来测试PC端的推理速度