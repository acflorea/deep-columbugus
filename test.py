from dbloader import getTextForDictionary, bugDicoToFullText, getBugDetails

# retrieve descriptions from db
# bug_descs = getTextForDictionary(1000)

# fulltextdesc = bugDicoToFullText(bug_descs)

# print fulltextdesc

bug_dataframe = getBugDetails('netbeansbugs')
# bug_dataframe = getBugDetails('eclipsebugs')
# bug_dataframe = getBugDetails('firefoxbugs_new')
print bug_dataframe.head(10)
