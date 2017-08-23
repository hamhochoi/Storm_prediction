import openpyxl
excel_document = openpyxl.load_workbook('storm.xlsx')

sheet_name = excel_document.get_sheet_names()
sheet = excel_document.get_sheet_by_name('Allstorms.ibtracs_wmo.v03r09')

all_rows = sheet.rows
all_columns = sheet.columns

row_count = sheet.max_row



# Split data into files:

#first_numb = all_columns[0][2]
#first_numb_val = str(first_numb.value)

count = 1;
str_count = str(count) + '.txt';

f = open(str_count, 'w');
f.write(str(all_columns[8][2].value));
f.write('\n');
f.write(str(all_columns[9][2].value));
f.write('\n');



for i in range(3, row_count):
	pre_numb = all_columns[0][i-1];
	now_numb = all_columns[0][i];
	pre_numb_val = str(pre_numb.value);
	now_numb_val = str(now_numb.value);
	
	if (now_numb_val == pre_numb_val):
		f.write(str(all_columns[8][i].value));
		f.write('\n');
		f.write(str(all_columns[9][i].value));
		f.write('\n');
	else:
		count = count + 1;
		str_count = str(count) + '.txt';
		f.close();
		f = open(str_count, 'w');
		
	if (i == row_count - 1):
		f.close();