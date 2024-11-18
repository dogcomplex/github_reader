
# https://chatgpt.com/share/673ad7b6-9044-8003-a1fd-46f5f6a3dc37
# https://github.com/nelsonic/github-scraper
# scrape via html instead of git api

# WIP template from o1

const gs = require('github-scraper');
const async = require('async');

// Define your project keywords
const keywords = 'machine learning';

// Construct the GitHub search URL
const searchUrl = `search?q=${encodeURIComponent(keywords)}&type=Repositories`;

// Step 1: Scrape the search results
gs(searchUrl, function(err, searchData) {
  if (err) {
    console.error(err);
    return;
  }

  // Assume searchData.entries contains repository URLs
  const repoUrls = searchData.entries.map(entry => entry.url);

  // Step 2: Collect repository details
  async.mapLimit(repoUrls, 5, (url, callback) => {
    gs(url, function(err, repoData) {
      if (err) {
        callback(err);
      } else {
        callback(null, repoData);
      }
    });
  }, function(err, reposData) {
    if (err) {
      console.error(err);
      return;
    }

    // Step 3: Analyze the data
    // For simplicity, let's match based on description keywords
    const similarRepos = reposData.filter(repo => {
      const description = repo.desc || '';
      return description.toLowerCase().includes('machine learning');
    });

    // Step 4: Output similar projects
    console.log('Similar Projects:', similarRepos);
  });
});
