document.addEventListener('DOMContentLoaded', function() {
  var footer = document.querySelector('footer');
  if (footer) {
    // Option 1: Find and remove the unwanted "Built with MkDocs" text.
    var builtWithText = footer.querySelector('.container .small'); // Usually the unwanted text is in this div
    if (builtWithText) {
      builtWithText.style.display = 'none'; // Hide the unwanted text
    }

    // Option 2: Insert logo in the footer
    var logo = document.createElement('img');
    logo.src = '/figures/logo.png';  // Path to the logo image
    logo.alt = 'EuBI Bridge logo';
    logo.style.maxWidth = '150px';  // Adjust size as needed
    logo.style.display = 'inline-block'; // Ensure logo stays inline

    // Append the logo to the footer
    var footerContent = footer.querySelector('.container');
    if (footerContent) {
      footerContent.appendChild(logo);  // Add logo next to existing footer content
    }
  }
});
