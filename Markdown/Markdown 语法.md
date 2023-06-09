# Markdown 语法
- [Markdown 语法](#markdown-语法)
- [Markdown 速查表](#markdown-速查表)
- [Markdown 标题语法](#markdown-标题语法)
  - [可选语法](#可选语法)
  - [标题（Headings）用法的最佳实践](#标题headings用法的最佳实践)
- [Markdown 段落](#markdown-段落)
  - [段落（Paragraph）用法的最佳实践](#段落paragraph用法的最佳实践)
- [Markdown 换行语法](#markdown-换行语法)
  - [换行（line-break）用法的最佳实践](#换行line-break用法的最佳实践)
- [Markdown 强调语法](#markdown-强调语法)
  - [粗体（Bold）](#粗体bold)
  - [粗体（Bold）用法最佳实践](#粗体bold用法最佳实践)
  - [斜体（Italic）](#斜体italic)
  - [斜体（Italic）用法的最佳实践](#斜体italic用法的最佳实践)
  - [粗体（Bold）和斜体（Italic）](#粗体bold和斜体italic)
  - [粗体（Bold）和斜体（Italic）用法的最佳实践](#粗体bold和斜体italic用法的最佳实践)
- [Markdown 引用语法](#markdown-引用语法)
  - [多个段落的块引用](#多个段落的块引用)
  - [嵌套块引用](#嵌套块引用)
  - [带有其它元素的块引用](#带有其它元素的块引用)
- [Markdown 列表语法](#markdown-列表语法)
  - [有序列表](#有序列表)
  - [有序列表最佳实践](#有序列表最佳实践)
  - [无序列表](#无序列表)
  - [无序列表最佳实践](#无序列表最佳实践)
  - [在列表中嵌套其他元素](#在列表中嵌套其他元素)
    - [段落](#段落)
    - [引用块](#引用块)
    - [代码块](#代码块)
    - [图片](#图片)
    - [列表](#列表)
- [Markdown 代码语法](#markdown-代码语法)
  - [转义反引号](#转义反引号)
  - [代码块](#代码块-1)
- [Markdown 分隔线语法](#markdown-分隔线语法)
  - [分隔线（Horizontal Rule）用法最佳实践](#分隔线horizontal-rule用法最佳实践)
- [Markdown 链接语法](#markdown-链接语法)
  - [给链接增加 Title](#给链接增加-title)
  - [网址和 Email 地址](#网址和-email-地址)
  - [带格式化的链接](#带格式化的链接)
  - [引用类型链接](#引用类型链接)
    - [链接的第一部分格式](#链接的第一部分格式)
    - [链接的第二部分格式](#链接的第二部分格式)
  - [链接最佳实践](#链接最佳实践)
- [Markdown 图片语法](#markdown-图片语法)
  - [链接图片](#链接图片)
  - [图片排版](#图片排版)
- [Markdown 转义字符语法](#markdown-转义字符语法)
  - [可做转义的字符](#可做转义的字符)
  - [特殊字符自动转义](#特殊字符自动转义)
- [Markdown 内嵌 HTML 标签](#markdown-内嵌-html-标签)
  - [行级內联标签](#行级內联标签)
  - [区块标签](#区块标签)
  - [HTML 用法最佳实践](#html-用法最佳实践)
- [Markdown 表格](#markdown-表格)
  - [对齐](#对齐)
  - [格式化表格中的文字](#格式化表格中的文字)
  - [在表中转义管道字符](#在表中转义管道字符)
- [Markdown 围栏代码块](#markdown-围栏代码块)
  - [语法高亮](#语法高亮)
- [Markdown 脚注](#markdown-脚注)
- [Markdown 标题编号](#markdown-标题编号)
  - [链接到标题 ID (#headid)](#链接到标题-id-headid)
    - [链接到标题 ID 最佳实践](#链接到标题-id-最佳实践)
- [Markdown 定义列表](#markdown-定义列表)
- [markdown 删除线](#markdown-删除线)
- [Markdown 任务列表语法](#markdown-任务列表语法)
- [Markdown 使用 Emoji 表情](#markdown-使用-emoji-表情)
  - [复制和粘贴表情符号](#复制和粘贴表情符号)
  - [使用表情符号简码](#使用表情符号简码)
- [自动网址链接](#自动网址链接)
  - [禁用自动 URL 链接](#禁用自动-url-链接)
- [Markdown 公式](#markdown-公式)

# Markdown 速查表
[Markdown 速查表](Markdown%20%E9%80%9F%E6%9F%A5%E8%A1%A8.md)

# Markdown 标题语法

要创建标题，请在单词或短语前面添加井号 (`#`) 。`#` 的数量代表了标题的级别。例如，添加三个 `#` 表示创建一个三级标题 (`<h3>`) (例如：`### My Header`)。

| Markdown 语法             | HTML                       |
| ------------------------ | -------------------------- |
| `# Heading level 1`      | `<h1>Heading level 1</h1>` |
| `## Heading level 2`     | `<h2>Heading level 2</h2>` |
| `### Heading level 3`    | `<h3>Heading level 3</h3>` |
| `#### Heading level 4`   | `<h4>Heading level 4</h4>` |
| `##### Heading level 5`  | `<h5>Heading level 5</h5>` |
| `###### Heading level 6` | `<h6>Heading level 6</h6>` |

## 可选语法

> 在`typora`和`Obsidian`都无法生效。

还可以在文本下方添加任意数量的 == 号来标识一级标题，或者 -- 号来标识二级标题。

| Markdown 语法                  | HTML                       |
| ----------------------------- | -------------------------- |
| `Heading level 1============` | `<h1>Heading level 1</h1>` |
| `Heading level 2------------` | `<h2>Heading level 2</h2>` |

## 标题（Headings）用法的最佳实践

不同的 Markdown 应用程序处理 `#` 和标题之间的空格方式并不一致。为了兼容考虑，请用一个空格在 `#` 和标题之间进行分隔。

| ✅ Do this            | ❌ Don't do this     |
| -------------------- | ------------------- |
| `# Here's a Heading` | `#Here's a Heading` |



# Markdown 段落

要创建段落，请使用空白行将一行或多行文本进行分隔。

| Markdown 语法                                                 | HTML                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| I really like using Markdown.<br />I think I'll use it to format all of my documents from now on. | <p>I really like using Markdown.</p><br /><p>I think I'll use it to format all of my documents from now on.</p> |

## 段落（Paragraph）用法的最佳实践

不要用空格（spaces）或制表符（tabs）缩进段落。

| ✅ Do this                                                    | ❌ Don't do this                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Don't put tabs or spaces in front of your paragraphs.<br />Keep lines left-aligned like this. | This can result in unexpected formatting problems. <br />  Don't add tabs or spaces in front of paragraphs. |

# Markdown 换行语法

在一行的末尾添加两个或多个空格，然后按回车键，即可创建一个换行 (`<br>`)。

| Markdown 语法                                            | HTML                                                         | 预览效果                                             |
| ------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| `This is the first line.  And this is the second line.` | `<p>This is the first line.<br>And this is the second line.</p>` | This is the first line. And this is the second line. |

## 换行（line-break）用法的最佳实践

> 在`typora`和`Obsidian`可以使用`shift+enter`来实现换行。

几乎每个 Markdown 应用程序都支持两个或多个空格进行换行，称为 `结尾空格（trailing whitespace)` 的方式，但这是有争议的，因为很难在编辑器中直接看到空格，并且很多人在每个句子后面都会有意或无意地添加两个空格。由于这个原因，你可能要使用除结尾空格以外的其它方式来换行。幸运的是，几乎每个 Markdown 应用程序都支持另一种换行方式：HTML 的 `<br>` 标签。

为了兼容性，请在行尾添加“结尾空格”或 HTML 的 `<br>` 标签来实现换行。

还有两种其他方式我并不推荐使用。CommonMark 和其它几种轻量级标记语言支持在行尾添加反斜杠 (`\`) 的方式实现换行，但是并非所有 Markdown 应用程序都支持此种方式，因此从兼容性的角度来看，不推荐使用。并且至少有两种轻量级标记语言支持无须在行尾添加任何内容，只须键入回车键（`return`）即可实现换行。

| ✅ Do this                                                    | ❌ Don't do this                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| First line with two spaces after.  And the next line.First line with the HTML tag after.<br>And the next line. | First line with a backslash after.\And the next line.<br />First line with nothing after.And the next line. |



# Markdown 强调语法

通过将文本设置为粗体或斜体来强调其重要性。

## 粗体（Bold）

要加粗文本，请在单词或短语的前后各添加两个星号（asterisks）或下划线（underscores）。如需加粗一个单词或短语的中间部分用以表示强调的话，请在要加粗部分的两侧各添加两个星号（asterisks）。

| Markdown 语法                 | HTML                                      | 预览效果                   |
| ---------------------------- | ----------------------------------------- | -------------------------- |
| `I just love **bold text**.` | `I just love <strong>bold text</strong>.` | I just love **bold text**. |
| `I just love __bold text__.` | `I just love <strong>bold text</strong>.` | I just love **bold text**. |
| `Love**is**bold`             | `Love<strong>is</strong>bold`             | Love**is**bold             |



## 粗体（Bold）用法最佳实践

Markdown 应用程序在如何处理单词或短语中间的下划线上并不一致。为兼容考虑，在单词或短语中间部分加粗的话，请使用星号（asterisks）。

| ✅ Do this        | ❌ Don't do this  |
| ---------------- | ---------------- |
| `Love**is**bold` | `Love__is__bold` |

## 斜体（Italic）

要用斜体显示文本，请在单词或短语前后添加一个星号（asterisk）或下划线（underscore）。要斜体突出单词的中间部分，请在字母前后各添加一个星号，中间不要带空格。

| Markdown 语法                           | HTML                                          | 预览效果                             |
| -------------------------------------- | --------------------------------------------- | ------------------------------------ |
| `Italicized text is the *cat's meow*.` | `Italicized text is the <em>cat's meow</em>.` | Italicized text is the *cat’s meow*. |
| `Italicized text is the _cat's meow_.` | `Italicized text is the <em>cat's meow</em>.` | Italicized text is the *cat’s meow*. |
| `A*cat*meow`                           | `A<em>cat</em>meow`                           | A*cat*meow                           |

## 斜体（Italic）用法的最佳实践

要同时用粗体和斜体突出显示文本，请在单词或短语的前后各添加三个星号或下划线。要加粗并用斜体显示单词或短语的中间部分，请在要突出显示的部分前后各添加三个星号，中间不要带空格。

| ✅ Do this    | ❌ Don't do this |
| ------------ | --------------- |
| `A*cat*meow` | `A_cat_meow`    |

## 粗体（Bold）和斜体（Italic）

> 一句话总结：粗体（Bold）和斜体（Italic）使用``三个星号``即可

要同时用粗体和斜体突出显示文本，请在单词或短语的前后各添加三个星号或下划线。要加粗并用斜体显示单词或短语的中间部分，请在要突出显示的部分前后各添加三个星号，中间不要带空格。

| Markdown 语法                              | HTML                                                         | 预览效果                                |
| ----------------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| `This text is ***really important***.`    | `This text is <strong><em>really important</em></strong>.`   | This text is ***really important***.    |
| `This text is ___really important___.`    | `This text is <strong><em>really important</em></strong>.`   | This text is ___really important___.    |
| `This text is __*really important*__.`    | `This text is <strong><em>really important</em></strong>.`   | This text is __*really important*__.    |
| `This text is **_really important_**.`    | `This text is <strong><em>really important</em></strong>.`   | This text is **_really important_**.    |
| `This is really***very***important text.` | `This is really<strong><em>very</em></strong>important text.` | This is really***very***important text. |

## 粗体（Bold）和斜体（Italic）用法的最佳实践

Markdown 应用程序在处理单词或短语中间添加的下划线上并不一致。为了实现兼容性，请使用星号将单词或短语的中间部分加粗并以斜体显示，以示重要。

| ✅ Do this                                 | ❌ Don't do this                           |
| ----------------------------------------- | ----------------------------------------- |
| `This is really***very***important text.` | `This is really___very___important text.` |



# Markdown 引用语法

要创建块引用，请在段落前添加一个 `>` 符号。

```text
> Dorothy followed her through many of the beautiful rooms in her castle.
```

渲染效果如下所示：

> Dorothy followed her through many of the beautiful rooms in her castle.

## 多个段落的块引用

块引用可以包含多个段落。为段落之间的空白行添加一个 `>` 符号。

```text
> Dorothy followed her through many of the beautiful rooms in her castle.
>
> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.
```

渲染效果如下：

> Dorothy followed her through many of the beautiful rooms in her castle.
>
> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

## 嵌套块引用

块引用可以嵌套。在要嵌套的段落前添加一个 `>>` 符号。

```text
> Dorothy followed her through many of the beautiful rooms in her castle.
>
>> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.
```

渲染效果如下：

> Dorothy followed her through many of the beautiful rooms in her castle.
>
> > The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

## 带有其它元素的块引用

块引用可以包含其他 Markdown 格式的元素。并非所有元素都可以使用，你需要进行实验以查看哪些元素有效。

```text
> #### The quarterly results look great!
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.
```

渲染效果如下：

> #### The quarterly results look great!
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
> *Everything* is going according to **plan**.



# Markdown 列表语法

可以将多个条目组织成有序或无序列表。

## 有序列表

要创建有序列表，请在每个列表项前添加数字并紧跟一个英文句点。数字不必按数学顺序排列，但是列表应当以数字 1 起始。

| Markdown 语法                                                 | HTML                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1. First item<br />2. Second item<br />3. Third item\|<br />4. Fourth item | <ol><br /><li>First item</li><br /><li>Second item</li><br /><li>Third item</li><br /><li>Fourth item</li><br /></ol> |
| 1. First item<br />2. Second item<br />3. Third item  <br />    1. Indented item<br />     2. Indented item<br />4. Fourth item | <ol><br /><li>First item</li><br /><li>Second item</li><br /><li>Third item<br /><ol><br /><li>Indented item</li><br /><li>Indented item</li><br /></ol><br /></li><br /><li>Fourth item</li><br /></ol> |

## 有序列表最佳实践

1. First item
2. Second item
3. Third item
   1. Indented item
   2. Indented item
4. Fourth item

## 无序列表

要创建无序列表，请在每个列表项前面添加破折号 (-)、星号 (*) 或加号 (+) 。缩进一个或多个列表项可创建嵌套列表。

| Markdown 语法                                                 | HTML                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| - First item<br />- Second item<br />- Third item<br />- Fourth item | <ul><br /><li>First item</li><br /><li>Second item</li><br /><li>Third item</li><br /><li>Fourth item</li><br /></ul> |
| * First item<br />* Second item<br />* Third item<br />* Fourth item | <ul><br /><li>First item</li><br /><li>Second item</li><br /><li>Third item</li><br /><li>Fourth item</li><br /></ul> |
| + First item<br />+ Second item<br />+ Third item<br />+ Fourth item | <ul><br /><li>First item</li><br /><li>Second item</li><br /><li>Third item</li><br /><li>Fourth item</li><br /></ul> |
| - First item<br />- Second item<br />- Third item<br />    - Indented item<br />    - Indented item<br />- Fourth item | <ul><br /><li>First item</li><br /><li>Second item</li><br /><li>Third item<br /><ul><br /><li>Indented item</li><br /><li>Indented item</li>\|<br /></ul><br /></li><br /><li>Fourth item</li><br /></ul> |

## 无序列表最佳实践

不要交替使用破折号 (-)、星号 (*) 或加号 (+)

- First item
- Second item
- Third item
  - ndented item
  - Indented item,
- Fourth item

## 在列表中嵌套其他元素

要在保留列表连续性的同时在列表中添加另一种元素，请将该元素缩进四个空格或一个制表符，如下例所示：

### 段落

```text
*   This is the first list item.
*   Here's the second list item.

    I need to add another paragraph below the second list item.

*   And here's the third list item.
```

渲染效果如下：

- This is the first list item.

- Here's the second list item.

  I need to add another paragraph below the second list item.

- And here's the third list item.

### 引用块

```text
*   This is the first list item.
*   Here's the second list item.

    > A blockquote would look great below the second list item.

*   And here's the third list item.
```

渲染效果如下：

- This is the first list item.

- Here's the second list item.

  > A blockquote would look great below the second list item.

- And here's the third list item.

### 代码块

代码块通常采用四个空格或一个制表符缩进。当它们被放在列表中时，请将它们缩进八个空格或两个制表符。

```text
1.  Open the file.
2.  Find the following code block on line 21:

        <html>
          <head>
            <title>Test</title>
          </head>

3.  Update the title to match the name of your website.
```

渲染效果如下：

1. Open the file.

2. Find the following code block on line 21:

   ```text
   <html>
     <head>
       <title>Test</title>
     </head>
   ```

3. Update the title to match the name of your website.

### 图片

```text
1.  Open the file containing the Linux mascot.
2.  Marvel at its beauty.

    ![Tux, the Linux mascot](/assets/images/tux.png)

3.  Close the file.
```

### 列表

You can nest an unordered list in an ordered list, or vice versa.

```text
1. First item
2. Second item
3. Third item
    - Indented item
    - Indented item
4. Fourth item
```

渲染效果如下：

1. First item
2. Second item
3. Third item
   - Indented item
   - Indented item
4. Fourth item



# Markdown 代码语法

要将单词或短语表示为代码，请将其包裹在反引号 ( ` ) 中。

| Markdown 语法                        | HTML                                           | 预览效果                            |
| ----------------------------------- | ---------------------------------------------- | ----------------------------------- |
| At the command prompt, type `nano`. | At the command prompt, type <code>nano</code>. | At the command prompt, type `nano`. |

## 转义反引号

如果你要表示为代码的单词或短语中包含一个或多个反引号，则可以通过将单词或短语包裹在双反引号 ( `` ) 中。

| Markdown 语法                          | HTML                                           | 预览效果                              |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------- |
| ``Use `code` in your Markdown file.`` | <code>Use `code` in your Markdown file.</code> | ``Use `code` in your Markdown file.`` |

## 代码块

要创建代码块，请将代码块的每一行缩进至少四个空格或一个制表符。

```text
    <html>
      <head>
      </head>
    </html>
```

渲染效果如下：

```text
<html>
  <head>
  </head>
</html>
```



# Markdown 分隔线语法

要创建分隔线，请在单独一行上使用三个或多个星号 (`***`)、破折号 (`---`) 或下划线 (`___`) ，并且不能包含其他内容。

```text
***

---

_________________
```

以上三个分隔线的渲染效果看起来都一样：

------

## 分隔线（Horizontal Rule）用法最佳实践

为了兼容性，请在分隔线的前后均添加空白行。

| ✅ Do this                                                    | ❌ Don't do this                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Try to put a blank line before...<br />---<br />...and after a horizontal rule. | Without blank lines, this would be a heading.<br />---<br />Don't do this! |



# Markdown 链接语法

链接文本放在中括号内，链接地址放在后面的括号中，链接 title 可选。

超链接 Markdown 语法代码：`[超链接显示名](超链接地址 "超链接title")`

对应的 HTML 代码：`<a href="超链接地址" title="超链接title">超链接显示名</a>`

```text
这是一个链接 [Markdown 语法](https://markdown.com.cn)。
```

渲染效果如下：

这是一个链接 [Markdown 语法](https://markdown.com.cn/)。

## 给链接增加 Title

链接 title 是当鼠标悬停在链接上时会出现的文字，这个 title 是可选的，它放在圆括号中链接地址后面，跟链接地址之间以空格分隔。

```text
这是一个链接 [Markdown 语法](https://markdown.com.cn "最好的markdown教程")。
```

渲染效果如下：

这是一个链接 [Markdown 语法](https://markdown.com.cn/)。

## 网址和 Email 地址

使用尖括号可以很方便地把 URL 或者 email 地址变成可点击的链接。

```text
<https://markdown.com.cn>
<fake@example.com>
```

渲染效果如下：

[https://markdown.com.cn](https://markdown.com.cn/)
[fake@example.com](mailto:fake@example.com)

## 带格式化的链接

[强调](https://markdown.com.cn/basic-syntax/links.html#emphasis) 链接，在链接语法前后增加星号。要将链接表示为代码，请在方括号中添加反引号。

```text
I love supporting the **[EFF](https://eff.org)**.
This is the *[Markdown Guide](https://www.markdownguide.org)*.
See the section on [`code`](#code).
```

渲染效果如下：

I love supporting the **[EFF](https://eff.org/)**.
This is the *[Markdown Guide](https://www.markdownguide.org/)*.
See the section on [`code`](https://markdown.com.cn/basic-syntax/links.html#code).

## 引用类型链接

引用样式链接是一种特殊的链接，它使 URL 在 Markdown 中更易于显示和阅读。参考样式链接分为两部分：与文本保持内联的部分以及存储在文件中其他位置的部分，以使文本易于阅读。

### 链接的第一部分格式

引用类型的链接的第一部分使用两组括号进行格式设置。第一组方括号包围应显示为链接的文本。第二组括号显示了一个标签，该标签用于指向您存储在文档其他位置的链接。

~~尽管不是必需的，可以在第一组和第二组括号之间包含一个空格~~。第二组括号中的标签不区分大小写，可以包含字母，数字，空格或标点符号。

以下示例格式对于链接的第一部分效果相同：

- [hobbit-hole][1]
- `[hobbit-hole][1]`

### 链接的第二部分格式

引用类型链接的第二部分使用以下属性设置格式：

1. 放在括号中的标签，其后紧跟一个冒号和至少一个空格（例如`[label]:`）。
2. 链接的 URL，可以选择将其括在尖括号中。
3. 链接的可选标题，可以将其括在双引号，单引号或括号中。

以下示例格式对于链接的第二部分效果相同：

- `[1]: https://en.wikipedia.org/wiki/Hobbit#Lifestyle`
- `[1]: https://en.wikipedia.org/wiki/Hobbit#Lifestyle "Hobbit lifestyles"`
- `[1]: https://en.wikipedia.org/wiki/Hobbit#Lifestyle 'Hobbit lifestyles'`
- `[1]: https://en.wikipedia.org/wiki/Hobbit#Lifestyle (Hobbit lifestyles)`
- `[1]: <https://en.wikipedia.org/wiki/Hobbit#Lifestyle> "Hobbit lifestyles"`
- `[1]: <https://en.wikipedia.org/wiki/Hobbit#Lifestyle> 'Hobbit lifestyles'`
- `[1]: <https://en.wikipedia.org/wiki/Hobbit#Lifestyle> (Hobbit lifestyles)`

可以将链接的第二部分放在 Markdown 文档中的任何位置。有些人将它们放在出现的段落之后，有些人则将它们放在文档的末尾（例如尾注或脚注）。

[1]:https://en.wikipedia.org/wiki/Hobbit#Lifestyle

## 链接最佳实践

不同的 Markdown 应用程序处理 URL 中间的空格方式不一样。为了兼容性，请尽量使用%20 代替空格。

| ✅ Do this                                           | ❌ Don't do this                                 |
| --------------------------------------------------- | ----------------------------------------------- |
| `[link](https://www.example.com/my%20great%20page)` | `[link](https://www.example.com/my great page)` |



# Markdown 图片语法

要添加图像，请使用感叹号 (`!`), 然后在方括号增加替代文本，图片链接放在圆括号里，括号里的链接后可以增加一个可选的图片标题文本。

插入图片 Markdown 语法代码：`![图片alt](图片链接 "图片title")`。

对应的 HTML 代码：`<img src="图片链接" alt="图片alt" title="图片title">`

```text
![这是图片](/assets/img/philly-magic-garden.jpg "Magic Gardens")
```

渲染效果如下：

![这是图片](Markdown 语法.assets/philly-magic-garden.9c0b4415.jpg)

## 链接图片

给图片增加链接，请将图像的 Markdown 括在方括号中，然后将链接添加在圆括号中。

```text
[![沙漠中的岩石图片](/assets/img/shiprock.jpg "Shiprock")](https://markdown.com.cn)
```

## 图片排版

将图片设置为左对齐

```
<img src = "./images/logo.png" align="left">
```

两张图片并排显示 - 设置相关的 width 即可

```
<center class="half">
    <img src="img1.jpg" width="300"/>
    <img src="img2.jpg" width="300"/>
</center>
```

将图片居中

```
<div align=center><img src="地址" style="zoom:10%;" /></div>
```



# Markdown 转义字符语法

要显示原本用于格式化 Markdown 文档的字符，请在字符前面添加反斜杠字符 \ 。

```text
\* Without the backslash, this would be a bullet in an unordered list.
```

渲染效果如下：

\* Without the backslash, this would be a bullet in an unordered list.

## 可做转义的字符

以下列出的字符都可以通过使用反斜杠字符从而达到转义目的。

| Character | Name                                                         |
| --------- | ------------------------------------------------------------ |
| \         | backslash                                                    |
| `         | backtick (see also [escaping backticks in code](https://markdown.com.cn/basic-syntax/escaping-characters.html#escaping-backticks)) |
| *         | asterisk                                                     |
| _         | underscore                                                   |
| { }       | curly braces                                                 |
| [ ]       | brackets                                                     |
| ( )       | parentheses                                                  |
| #         | pound sign                                                   |
| +         | plus sign                                                    |
| -         | minus sign (hyphen)                                          |
| .         | dot                                                          |
| !         | exclamation mark                                             |
| \|        | pipe (see also [escaping pipe in tables](https://markdown.com.cn/extended-syntax/escaping-pipe-characters-in-tables.html)) |

## 特殊字符自动转义

在 HTML 文件中，有两个字符需要特殊处理： `<` 和 `&` 。 `<` 符号用于起始标签，`&` 符号则用于标记 HTML 实体，如果你只是想要使用这些符号，你必须要使用实体的形式，像是 `<` 和 `&`。

`&` 符号其实很容易让写作网页文件的人感到困扰，如果你要打「AT&T」 ，你必须要写成「`AT&T`」 ，还得转换网址内的 `&` 符号，如果你要链接到：

```
http://images.google.com/images?num=30&q=larry+bird
```

你必须要把网址转成：

```
http://images.google.com/images?num=30&amp;q=larry+bird
```

才能放到链接标签的 `href` 属性里。不用说也知道这很容易忘记，这也可能是 HTML 标准检查所检查到的错误中，数量最多的。

Markdown 允许你直接使用这些符号，它帮你自动转义字符。如果你使用 `&` 符号的作为 HTML 实体的一部分，那么它不会被转换，而在其它情况下，它则会被转换成 `&`。所以你如果要在文件中插入一个著作权的符号，你可以这样写：

```
&copy;
```

Markdown 将不会对这段文字做修改，但是如果你这样写：

```
AT&T
```

Markdown 就会将它转为：

```
AT&amp;T
```

类似的状况也会发生在 `<` 符号上，因为 Markdown 支持 [行内 HTML](https://markdown.com.cn/basic-syntax/#内联-html) ，如果你使用 `<` 符号作为 HTML 标签的分隔符，那 Markdown 也不会对它做任何转换，但是如果你是写：

```
4 < 5
```

Markdown 将会把它转换为：

```
4 &lt; 5
```

需要特别注意的是，在 Markdown 的块级元素和内联元素中， `<` 和 `&` 两个符号都会被自动转换成 HTML 实体，这项特性让你可以很容易地用 Markdown 写 HTML。（在 HTML 语法中，你要手动把所有的 `<` 和 `&` 都转换为 HTML 实体。）



# Markdown 内嵌 HTML 标签

对于 Markdown 涵盖范围之外的标签，都可以直接在文件里面用 HTML 本身。如需使用 HTML，不需要额外标注这是 HTML 或是 Markdown，只需 HTML 标签添加到 Markdown 文本中即可。

## 行级內联标签

HTML 的行级內联标签如 `<span>`、`<cite>`、`<del>` 不受限制，可以在 Markdown 的段落、列表或是标题里任意使用。依照个人习惯，甚至可以不用 Markdown 格式，而采用 HTML 标签来格式化。例如：如果比较喜欢 HTML 的 `<a>` 或 `<img>` 标签，可以直接使用这些标签，而不用 Markdown 提供的链接或是图片语法。当你需要更改元素的属性时（例如为文本指定颜色或更改图像的宽度），使用 HTML 标签更方便些。

HTML 行级內联标签和区块标签不同，在內联标签的范围内，Markdown 的语法是可以解析的。

```text
This **word** is bold. This <em>word</em> is italic.
```

渲染效果如下：

This **word** is bold. This *word* is italic.

## 区块标签

区块元素──比如 `<div>`、`<table>`、`<pre>`、`<p>` 等标签，必须在前后加上空行，以便于内容区分。而且这些元素的开始与结尾标签，不可以用 tab 或是空白来缩进。Markdown 会自动识别这区块元素，避免在区块标签前后加上没有必要的 `<p>` 标签。

例如，在 Markdown 文件里加上一段 HTML 表格：

```
This is a regular paragraph.

<table>
    <tr>
        <td>Foo</td>
    </tr>
</table>

This is another regular paragraph.
```

请注意，Markdown 语法在 HTML 区块标签中将不会被进行处理。例如，你无法在 HTML 区块内使用 Markdown 形式的`*强调*`。

## HTML 用法最佳实践

出于安全原因，并非所有 Markdown 应用程序都支持在 Markdown 文档中添加 HTML。如有疑问，请查看相应 Markdown 应用程序的手册。某些应用程序只支持 HTML 标签的子集。

对于 HTML 的块级元素 `<div>`、`<table>`、`<pre>` 和 `<p>`，请在其前后使用空行（blank lines）与其它内容进行分隔。尽量不要使用制表符（tabs）或空格（spaces）对 HTML 标签做缩进，否则将影响格式。

在 HTML 块级标签内不能使用 Markdown 语法。例如 `<p>italic and **bold**</p>` 将不起作用。



# Markdown 表格

要添加表，请使用三个或多个连字符（`---`）创建每列的标题，并使用管道（`|`）分隔每列。您可以选择在表的任一端添加管道。

```text
| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |
```

呈现的输出如下所示：

| Syntax    | Description |
| --------- | ----------- |
| Header    | Title       |
| Paragraph | Text        |

单元格宽度可以变化，如下所示。呈现的输出将看起来相同。

```text
| Syntax | Description |
| --- | ----------- |
| Header | Title |
| Paragraph | Text |
```

**Tip:** 使用连字符和管道创建表可能很麻烦。为了加快该过程，请尝试使用[Markdown Tables Generator](https://www.tablesgenerator.com/markdown_tables)。使用图形界面构建表，然后将生成的 Markdown 格式的文本复制到文件中。

## 对齐

您可以通过在标题行中的连字符的左侧，右侧或两侧添加冒号（`:`），将列中的文本对齐到左侧，右侧或中心。

```text
| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |
```

呈现的输出如下所示：

| Syntax    | Description |   Test Text |
| :-------- | :---------: | ----------: |
| Header    |    Title    | Here’s this |
| Paragraph |    Text     |    And more |

## 格式化表格中的文字

您可以在表格中设置文本格式。例如，您可以添加链接，代码（仅反引号（```）中的单词或短语，而不是代码块）和强调。

您不能添加标题，块引用，列表，水平规则，图像或 HTML 标签。

## 在表中转义管道字符

您可以使用表格的 HTML 字符代码（`|`）在表中显示竖线（`|`）字符。



# Markdown 围栏代码块

Markdown 基本语法允许您通过将行缩进四个空格或一个制表符来创建代码块。如果发现不方便，请尝试使用受保护的代码块。根据 Markdown 处理器或编辑器的不同，您将在代码块之前和之后的行上使用三个反引号（(`````）或三个波浪号（~~~）。

~~~text
```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```
~~~

呈现的输出如下所示：

```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```

**Tip:**要在代码块中显示反引号？请参阅了解如何[转义](#Markdown 转义字符语法)它们。

## 语法高亮

许多 Markdown 处理器都支持受围栏代码块的语法突出显示。使用此功能，您可以为编写代码的任何语言添加颜色突出显示。要添加语法突出显示，请在受防护的代码块之前的反引号旁边指定一种语言。

~~~text
```json
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```
~~~

呈现的输出如下所示：

```json
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```



# Markdown 脚注

脚注使您可以添加注释和参考，而不会使文档正文混乱。当您创建脚注时，带有脚注的上标数字会出现在您添加脚注参考的位置。读者可以单击链接以跳至页面底部的脚注内容。

要创建脚注参考，请在方括号（`[^1]`）内添加插入符号和标识符。标识符可以是数字或单词，但不能包含空格或制表符。标识符仅将脚注参考与脚注本身相关联 - 在输出中，脚注按顺序编号。

在括号内使用另一个插入符号和数字添加脚注，并用冒号和文本（`[^1]: My footnote.`）。您不必在文档末尾添加脚注。您可以将它们放在除列表，块引号和表之类的其他元素之外的任何位置。

```text
Here's a simple footnote,[^1] and here's a longer one.[^bignote]

[^1]: This is the first footnote.

[^bignote]: Here's one with multiple paragraphs and code.

    Indent paragraphs to include them in the footnote.

    `{ my code }`

    Add as many paragraphs as you like.
```

呈现的输出如下所示：

Here's a simple footnote,[^1] and here's a longer one.[^bignote]

[^1]: This is the first footnote.
[^bignote]: Here's one with multiple paragraphs and code.

```
Indent paragraphs to include them in the footnote.

`{ my code }`

Add as many paragraphs as you like.
```



# Markdown 标题编号

> `typora`貌似不支持

许多 Markdown 处理器支持[标题](https://markdown.com.cn/basic-syntax/headings.html)的自定义 ID - 一些 Markdown 处理器会自动添加它们。添加自定义 ID 允许您直接链接到标题并使用 CSS 对其进行修改。要添加自定义标题 ID，请在与标题相同的行上用大括号括起该自定义 ID。

```text
### My Great Heading {#custom-id}
```

HTML 看起来像这样：

```html
<h3 id="custom-id">My Great Heading</h3>
```

## 链接到标题 ID (#headid)

通过创建带有数字符号（`#`）和自定义标题 ID 的[标准链接]((/basic-syntax/links.html)，可以链接到文件中具有自定义 ID 的标题。

| Markdown                      | HTML                                     |
| ----------------------------- | ---------------------------------------- |
| `[Heading IDs](#heading-ids)` | `<a href="#heading-ids">Heading IDs</a>` |

其他网站可以通过将自定义标题 ID 添加到网页的完整 URL（例如`[Heading IDs](https://markdown.com.cn/extended-syntax/heading-ids.html#headid)`）来链接到标题。

### 链接到标题 ID 最佳实践

[回到顶部](#Markdown 标题语法)



# Markdown 定义列表

一些 Markdown 处理器允许您创建术语及其对应定义的*定义列表*。要创建定义列表，请在第一行上键入术语。在下一行，键入一个冒号，后跟一个空格和定义。

```text
First Term
: This is the definition of the first term.

Second Term
: This is one definition of the second term.
: This is another definition of the second term.
```

HTML 看起来像这样：

```html
<dl>
  <dt>First Term</dt>
  <dd>This is the definition of the first term.</dd>
  <dt>Second Term</dt>
  <dd>This is one definition of the second term. </dd>
  <dd>This is another definition of the second term.</dd>
</dl>
```

呈现的输出如下所示：

- First Term

  This is the definition of the first term.

- Second Term

  This is one definition of the second term.

  This is another definition of the second term.



# markdown 删除线

您可以通过在单词中心放置一条水平线来删除单词。结果看起来像这样。此功能使您可以指示某些单词是一个错误，要从文档中删除。若要删除单词，请在单词前后使用两个波浪号`~~`。

```text
~~世界是平坦的。~~ 我们现在知道世界是圆的。
```

呈现的输出如下所示：

世界是平坦的。我们现在知道世界是圆的。



# Markdown 任务列表语法

任务列表使您可以创建带有复选框的项目列表。在支持任务列表的 Markdown 应用程序中，复选框将显示在内容旁边。要创建任务列表，请在任务列表项之前添加破折号`-`和方括号`[ ]`，并在`[ ]`前面加上空格。要选择一个复选框，请在方括号`[x]`之间添加 x。

```text
- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media
```

呈现的输出如下所示：

- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media



# Markdown 使用 Emoji 表情

有两种方法可以将表情符号添加到 Markdown 文件中：将表情符号复制并粘贴到 Markdown 格式的文本中，或者键入*emoji shortcodes*。

## 复制和粘贴表情符号

许多 Markdown 应用程序会自动以 Markdown 格式的文本显示表情符号。从 Markdown 应用程序导出的 HTML 和 PDF 文件应显示表情符号。

**Tip:** 如果您使用的是静态网站生成器，请确保将 HTML 页面编码为 UTF-8。.

## 使用表情符号简码

一些 Markdown 应用程序允许您通过键入表情符号短代码来插入表情符号。这些以冒号开头和结尾，并包含表情符号的名称。

```text
去露营了！ :tent: 很快回来。

真好笑！ :joy:
```

呈现的输出如下所示：

去露营了！⛺很快回来。

真好笑！😂

**Note:** 注意：您可以使用此[表情符号简码列表](https://gist.github.com/rxaviers/7360908)，但请记住，表情符号简码因应用程序而异。



# 自动网址链接

许多 Markdown 处理器会自动将 URL 转换为链接。这意味着如果您输入 http://www.example.com，即使您未[使用方括号](https://markdown.com.cn/basic-syntax/links.html)，您的 Markdown 处理器也会自动将其转换为链接。

```text
http://www.example.com
```

呈现的输出如下所示：

[http://www.example.com](http://www.example.com/)

## 禁用自动 URL 链接

如果您不希望自动链接 URL，则可以通过将 URL 表示为带反引号的代码来删除该链接。

```text
`http://www.example.com`
```

呈现的输出如下所示：

```
http://www.example.com
```

# Markdown 公式
[Markdown 公式](Markdown%20%E5%85%AC%E5%BC%8F.md)





















