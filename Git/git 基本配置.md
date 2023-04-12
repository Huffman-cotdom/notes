# Git基本配置

## 安装

```shell
brew install git # MacOS
```

确认是否安装成功

```shell
$ git --version
git version 2.32.0 (Apple Git-132)
```

## 配置用户信息

```shell
git config --global user.name "runoob"
git config --global user.email test@runoob.com
```

## 设置全局 .gitignore_global

> 可选

使用`gi`工具构建`.gitignore_global`

### 安装 gi 工具

**zsh**

```shell
echo "function gi() { curl -sLw "\n" https://www.toptal.com/developers/gitignore/api/\$@ ;}" >> ~/.zshrc && source ~/.zshrc
```

查看是否安装成功并打印所有可忽略名单

```shell
gi list
```

### 创建.gitignore_global

```shell
gi macos,visualstudiocode >> ~/.gitignore_global
```

将`.gitignore_global`文件添加到`~/.gitconfig`中

```shell
[user]
	name = fulei
	email = candy.fulei@gmail.com
[core]
	excludesfile = ~/.gitignore_global
```

### raycast gi 工具

![image-20230413002349309](./git 基本配置.assets/image-20230413002349309.png)

## 连接本地与远程服务器连接

### 创建远程仓库

在远程服务器（GitHub、GitLab 等）创建仓库。

### 连接远程仓库

```shell
ssh-keygen -t rsa -C "youremail@example.com"
```

将生成的密钥：`~/.ssh/id_rsa.pub`中的内容复制到远程服务器（比如：GItHub）中的SSH-keys中。

验证是否成功

```shell
$ ssh -T git@github.com
The authenticity of host 'github.com (20.205.243.166)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
Hi Huffman-cotdom! You've successfully authenticated, but GitHub does not provide shell access.
```

与服务器仓库建立连接

```shell
git remote add origin git@github.com:Huffman-cotdom/Technology_reserve_box.git
```

切换分支

```shell
git branch -M main
```

将本地仓库文件提交到暂存区

```shell
git add .
```

```shell
git status
```

执行提交操作

```shell
git commit -m "init"
```

将文件push到远程服务器仓库

```shell
git push -u origin main
```