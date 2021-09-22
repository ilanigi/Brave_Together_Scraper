"""
Hello
Odysee.com is a video streaming site that believes in free speech and therfore doesn't censor holocaust denial
For the purpose of creating an AI that can detect holocaust denial posts, I found the site usefull for enlarging the AI's initial database. 
(Many posts in the web are just the names of videos with a link to watch them..)
I created this scraper using the selenium module to fetch video names that result when searching for holocaust denial content in the site.
The results are saved in a csv file.
"""

from selenium import webdriver
import time
import pandas as pd

#This is the function for scraping the site
def scraper_test(search_term,file_path,limit):
    #The Url for serching. The search term entered by user will be added to the url to find its results
    url="https://odysee.com/$/search?q={}".format(search_term)
    #open chrome with web driver. Driver can be downloaded at: https://chromedriver.chromium.org/downloads
    driver=webdriver.Chrome(executable_path=r'{}'.format(file_path))
    #open site
    driver.get(url)
    #Pause the program so the site can load
    time.sleep(40)
    #the list where the results will go to
    lis=list()
    #Variable for counting each result
    i=1
    #scrape each result of the search
    while i<limit:
        #find the html element with the results
         result_list_element=driver.find_elements_by_xpath('//*[@id="main-content"]/section/section/ul/li[{}]/div/div/div[1]/div[1]/a/div/span'.format(i))
         #To make sure the program works optimally it's good to pause it a bit
         time.sleep(2)
         #The elemnt is initialy a list. Here we will fetch the text you want from the element and add it to the list
         for result_element in result_list_element:
             text_of_element=result_element.get_attribute('textContent')
             lis.append(text_of_element)
             #I like to print each result to see that my program's preforming well
             print(text_of_element)
             #After 19 results we will have to scroll down the window to get more results
         if i>19:
            #Scroll down on the web browser
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(5)
         #Update count of results    
         i=i+1
         print(i)
    #close web -browser
    driver.quit()
    #print the list to see if it came out good 
    print(lis)
    #Import the results to a csv file
    list_to_csv(lis)

#Import the results into a csv file
def list_to_csv(lis):
    dict={"Content":lis}
    #create data-frame based on the list
    df = pd.DataFrame(dict)
    time.sleep(1)
    #add dataframe with the results into a csv file
    #The file will be saved in a new csv. 
    file_name=input('Please enter file name for results to be sved in: ')
    df.to_csv('{}.csv'.format(file_name))

# Defining main function
def main():
    search_term=input('Please enter a search term: ')
    #Enter the file path for the chrome web driver. Driver can be downloaded at: https://chromedriver.chromium.org/downloads
    file_path=input('Please enter the path for the chrome web driver: ')
    limit_input=input('Please enter the maximum amount of results you want: ')
    limit=int(limit_input)
    scraper_test(search_term,file_path,limit)
# Using the special variable 
# __name__
if __name__=="__main__":
    main()
