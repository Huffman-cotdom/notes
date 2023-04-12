# Git中Tag相关操作

git tag -a <tagname> 将会给最近提交的一次 commit 打上 tag 标签，然后使用 git push 提交该 tag。

```bash
git tag -a <tagname> -m "tag说明"
git push -u origin <tagname>
```

查看所有 tag

```b
git tag
```

如果我们忘了给某个提交打标签，又将它发布了，我们可以给它追加标签

```bash
$ git tag -a v0.9 85fc7e7
$ git log --oneline --decorate --graph
*   d5e9fc2 (HEAD -> master) Merge branch 'change_site'
|\  
| * 7774248 (change_site) changed the runoob.php
* | c68142b 修改代码
|/  
* c1501a2 removed test.txt、add runoob.php
* 3e92c19 add test.txt
* 3b58100 (tag: v0.9) 第一次版本提交
```

删除本地的 tag

```
git tag -d <tagname>
```

删除远程的 tag

```bash
git push origin --delete tag <tagname>
```

查看此 tag 所有的修改信息

```bash
git show v1.0
```

修改 tag 名称

将v6.2.0重命名为v6.6.2

- 新版本号：v6.6.2
- 错误版本号：v6.2.0

```bash
git tag v6.6.2 v6.2.0
```