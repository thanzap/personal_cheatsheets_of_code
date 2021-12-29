#round a cell
=round(cell)
#square a cell
=sqrt(cell)
#combine 2 functions 
=round(sqrt(cell))
#min
=min(cell:cell) #range of cells
#max
=max(cell:cell) #range of cells
#sum
=sum(cell:cell) #range of cells
#average-mean
=average(cell:cell) #range of cells
#median
=median(cell:cell) #range of cells
#rank of a value in a column in a range of cells
=rank(cell,$column$cell:$column$cell)
#rank of a value in a column in a range of cells in ascending order
=rank(cell,$column$cell:$column$cell , 1) #default is 0 which can also be absent which means descending
#worst 2 values by ranking
=cell <= 2
#string manipulation - extract part of a string using Left from the start of string or RIGHT from the end of string
=RIGHT(cell, 4) #4 is the number of the characters we want to keep
=LEFT(cell, 4) #4 is the number of the characters we want to keep
#length of a string in a cell
=LENGTH(cell)
#search for a character or space in a cell
=search("character", cell)
#take the length of the remaining string after the space or character
=cell- cell #1st cell is the length of string - 2nd cell is the position that the space or character occurs
#get the remaining string after the space or character e.g. example
=RIGHT(cell, cell) #1st cell is the string, 2nd cell is the calculated length of the remaing string after the space or character
#example
= RIGHT(cell, LEN(cell) - SEARCH(" ", cell)) #only keeping the 2nd word from a cell (the words are separated by a space " ")
#concatenate a string 
=CONCATENATE(cell ," ",cell) #cells contain string
#find the weekday
=WEEKDAY(cell,type) #cell which contains date, type
#type = 1: Sunday is day 1 and Saturday is day 7 (default) 
#type = 2: Monday is day 1 and Sunday is day 7
#type = 3: Monday is day 0 and Sunday is day 6
#check if the weekday is friday
=cell=4 #weekday cell, 4 which means friday
#difference between 2 dates 
=datedif(start_date, end_date, unit) #end date or alternative NOW(), unit can be "Y","M","D" (year month day)
#flow control-IF
=IF(logical expression,value if true,value if false)
== #would evaluate 2
#nested logical functions- If
IF(H14 < ___, "bad", IF(H14 < ___, "acceptable", "perfect"))
#Combining logical values - OR, WEEKDAY
=OR(WEEKDAY(B27,2)=6,WEEKDAY(B27,2)=7) #checks if the date is saturday or sunday and returns TRUE OR FALSE
#Conditional counting - COUNTIF
=COUNTIF(range, criterion) #count the number of times the criterion is met in the specified range
#Conditional sum - SUMIF
=SUMIF(range, criterion, sum_range) #evaluates to the conditional sum across a range.
#Conditional average - AVERAGEIF
=AVERAGEIF(range, criterion, average_range)  #evaluates to the conditional average across a range.
=AVERAGEIF($B$3:$B$26,"<= 2017-07-01",$D$3:$D$26) #get the average of a range before a date
#we want to calculate the average amount spent on dinners
=AVERAGE(FILTER(D3:D26, E3:E26 = "Dinner"))
#get the median from a range of cells before a date
=MEDIAN(FILTER(D3:D26, ___ <= DATEVALUE(___)))
#Automating the lookup - VLOOKUP
=VLOOKUP(search_key, range, index, is_sorted) #look for a match in the leftmost column of a lookup table and return the value 
#in a certain column
#You can compare it to the process of looking through a phone book. The search_key would be the name of the person you want the phone number of. The range is the data in the book, 
#with the names in the leftmost column. Finally, the index is the number of the column where you find what you need, the phone number.
#Horizontal lookup - HLOOKUP
=HLOOKUP(search_key, range, index, is_sorted)
#Weighted average - SUMPRODUCT figure out the sum of products of 2 or more ranges of equal size.
=SUMPRODUCT(array1, [array2, ...]) 
#What IS*() the data type?
=ISTEXT(cell)
=ISNUMBER(cell)
=ISDATE(cell)
=ISLOGICAL(cell)
=ISURL(cell)
=HYPERLINK[url]
=ISFORMULA(cell)
#Hyperlink are not url but they are formulas
=ISBLANK(cell)
#get the rows with a blank value from a range of A2:H20
=FILTER(A2:H20,H2:H20) #H2:H20 is the logical value which shows if ISBLANK or not.
#CONVERT DATA TYPES
=N(cell) #convert it to number type
=TO_PERCENT() #CONVERT TO PERCENT TYPE
#=cell/absolutevalueofcellwithbesttime returns the ratio of best time
=A2/$A$20 #ratio of best time which is in cell A20
#convert time and speed
=CONVERT(1, "hr", "sec") #RETURNS 3600
=CONVERT(1.4,"m/s","mph") #Returns 3.131















