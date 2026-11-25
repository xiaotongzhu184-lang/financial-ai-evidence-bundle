// Optional rendered regression: NODE_PATH=<playwright packages> node this-file URL
// Run against a verified real-source review server containing long cited tokens.
const { chromium } = require('playwright');

(async () => {
  const url = process.argv[2];
  if (!url || !/^http:\/\/127\.0\.0\.1:\d+\/$/.test(url)) {
    throw new Error('Provide the exact loopback research review URL, ending in /.');
  }
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    await page.goto(url, { timeout: 120000 });
    await page.waitForLoadState('networkidle');
    const result = await page.evaluate(() => {
      const table = document.querySelector('.table');
      const quotes = Array.from(document.querySelectorAll('blockquote'));
      return {
        title: document.title,
        viewport: innerWidth,
        documentWidth: document.documentElement.scrollWidth,
        longSourceTokenPresent: quotes.some(el => /\S{48,}/.test(el.textContent)),
        citationOverflow: quotes.some(el => el.scrollWidth > el.clientWidth + 1),
        tableScrollWidth: table?.scrollWidth,
        tableClientWidth: table?.clientWidth,
        tableOverflowX: table ? getComputedStyle(table).overflowX : null,
      };
    });
    console.log(JSON.stringify(result));
    if (!result.longSourceTokenPresent) throw new Error('This regression requires an actual long source citation token.');
    if (result.documentWidth > result.viewport || result.citationOverflow) throw new Error('Source citations overflow the mobile page.');
    if (!(result.tableScrollWidth > result.tableClientWidth) || result.tableOverflowX !== 'auto') {
      throw new Error('The wide metric table must retain its internal horizontal scrolling.');
    }
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
