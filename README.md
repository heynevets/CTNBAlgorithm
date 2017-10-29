# CTNBAlgorithm

Lets use the IEEE journal template for this project

## Guidelines

I'm posting the guidelines from Prof's website:

1. title page, preface, acknowledgements, table of content, list of tables/figures, abstract
2. introduction
3. theoretical bases and literature review
4. hypothesis (or goals)
5. methodology
6. implementation
7. data analysis and discussion
8. conclusions and recommendations
9. bibliography
10. appendices


## Directories and files

The folder /template has the original IEEE template I downloaded
/Working is the directory where we store all our works

## File Arrangement

Every section has its own separate file we can work on.
Only the abstract and bibliography are in the main.tex file.

### Abstract
Abstract is written in the main.tex file.
Somehow I could not separate it to a different file.


### Citations
To use citation in the work, just add the bibliography in the main.tex file. 


The bibliography section is at the very end.
Each entry starts with the line \bibitem{T1_Paper}
In this case, T1_Paper is the tag you can use in the content as:

\cite{T1_Paper}

The compiler will automatically generate corresponding [#] with the command


