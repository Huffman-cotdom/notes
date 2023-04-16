# Dockerfile

DockerFile 是用来构建 docker 镜像的文件的命令参数脚本

构建步骤：

1. 编写一个 dockerfile 脚本
2. docker build 构建成为一个镜像
3. docker run 运行镜像
4. docker push 发布镜像 (Docker hub , 阿里云镜像仓库！)

## Dockerfile 命令

https://yeasy.gitbook.io/docker_practice/image/dockerfile

```shell
FROM # 基础镜像, 一切从这里开始构建
MANTAINER # 镜像是谁写的, 姓名+邮箱
RUN # 镜像构建的时候需要运行的命令
COPY # 类似ADD,将我们文件拷贝到镜像中
ADD # 更高级的COPY命令
WORKDIR # 镜像的工作目录
VOLUME # 挂载的目录
EXPOSE # 暴露端口配置
RUN # 运行
CMD # 指定这个容器启动的时候要运行的命令,只有最后一个会生效,可被替代
ENTRYPOINT # 指定这个容器启动的时候要运行的命令,可以追加命令
ONBUILD # 当构建一个被继承 DockerFile 这个时候就会运行ONBUILD的指令,触发指令
ENV # 构建的时候设置环境变量!
```