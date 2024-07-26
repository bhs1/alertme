const customCSS = `
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #27272a;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 0.375rem;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
`;

const styleTag = document.createElement("style");
styleTag.textContent = customCSS;
document.head.append(styleTag);

let labels = [];

function unmarkPage() {
  // Unmark page logic
  for (const label of labels) {
    document.body.removeChild(label);
  }
  labels = [];
}

function markPage() {
  unmarkPage();

  var bodyRect = document.body.getBoundingClientRect();

  var items = Array.prototype.slice
    .call(document.querySelectorAll("*"))
    .map(function (element) {
      var vw = Math.max(
        document.documentElement.clientWidth || 0,
        window.innerWidth || 0
      );
      var vh = Math.max(
        document.documentElement.clientHeight || 0,
        window.innerHeight || 0
      );
      var textualContent = element.textContent.trim().replace(/\s{2,}/g, " ");
      var elementType = element.tagName.toLowerCase();
      var ariaLabel = element.getAttribute("aria-label") || "";

      var rects = [...element.getClientRects()]
        .filter((bb) => {
          var center_x = bb.left + bb.width / 2;
          var center_y = bb.top + bb.height / 2;
          var elAtCenter = document.elementFromPoint(center_x, center_y);

          return elAtCenter === element || element.contains(elAtCenter);
        })
        .map((bb) => {
          const rect = {
            left: Math.max(0, bb.left),
            top: Math.max(0, bb.top),
            right: Math.min(vw, bb.right),
            bottom: Math.min(vh, bb.bottom),
          };
          return {
            ...rect,
            width: rect.right - rect.left,
            height: rect.bottom - rect.top,
          };
        });

      var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

      // Check if the element is within a scrollable area
      function isWithinScrollableArea(element) {
        let parent = element.parentElement;
        while (parent) {
          if (
            parent.scrollHeight > parent.clientHeight ||
            parent.scrollWidth > parent.clientWidth
          ) {
            const overflowY = window.getComputedStyle(parent).overflowY;
            const overflowX = window.getComputedStyle(parent).overflowX;
            if (
              overflowY === "auto" ||
              overflowY === "scroll" ||
              overflowX === "auto" ||
              overflowX === "scroll"
            ) {
              return true;
            }
          }
          parent = parent.parentElement;
        }
        return false;
      }

      var isScrollableArea = isWithinScrollableArea(element);

      return {
        element: element,
        include:
          element.tagName === "INPUT" ||
          element.tagName === "TEXTAREA" ||
          element.tagName === "SELECT" ||
          element.tagName === "BUTTON" ||
          element.tagName === "A" ||
          element.onclick != null ||
          window.getComputedStyle(element).cursor == "pointer" ||
          element.tagName === "IFRAME" ||
          element.tagName === "VIDEO" ||
          isScrollableArea, // Include elements within scrollable areas
        area,
        rects,
        text: textualContent,
        type: elementType,
        ariaLabel: ariaLabel,
        isScrollableArea: isScrollableArea // Add this property to the item
      };
    })
    .filter((item) => item.include && item.area >= 20);

  // Only keep inner clickable items
  items = items.filter(
    (x) => !items.some((y) => x.element.contains(y.element) && !(x == y))
  );

  // Separate scrollable areas
  const scrollableAreas = items.filter(item => item.isScrollableArea);
  const nonScrollableItems = items.filter(item => !item.isScrollableArea);

  // Function to generate random colors
  function getRandomColor() {
    var letters = "0123456789ABCDEF";
    var color = "#";
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  // Lets create a floating border on top of these elements that will always be visible
  nonScrollableItems.forEach(function (item, index) {
    item.rects.forEach((bbox) => {
      newElement = document.createElement("div");
      var borderColor = getRandomColor();
      newElement.style.outline = `2px dashed ${borderColor}`;
      newElement.style.position = "fixed";
      newElement.style.left = bbox.left + "px";
      newElement.style.top = bbox.top + "px";
      newElement.style.width = bbox.width + "px";
      newElement.style.height = bbox.height + "px";
      newElement.style.pointerEvents = "none";
      newElement.style.boxSizing = "border-box";
      newElement.style.zIndex = 2147483647;

      // Add floating label at the bottom right corner
      var label = document.createElement("span");
      label.textContent = index;
      label.style.position = "absolute";
      // Adjusted to be at the bottom right corner
      label.style.bottom = "-18px";
      label.style.right = "-20px";
      label.style.background = borderColor;
      label.style.color = "white";
      label.style.padding = "2px 4px";
      label.style.fontSize = "12px";
      label.style.borderRadius = "2px";
      label.style.opacity = "0.5"; // Make it translucent
      newElement.appendChild(label);

      document.body.appendChild(newElement);
      labels.push(newElement);
    });
  });

  // Add bounding boxes for scrollable areas with unique numbers
  scrollableAreas.forEach(function (item, index) {
    const scrollableRect = item.rects.reduce((acc, rect) => {
      return {
        left: Math.min(acc.left, rect.left),
        top: Math.min(acc.top, rect.top),
        right: Math.max(acc.right, rect.right),
        bottom: Math.max(acc.bottom, rect.bottom),
        width: Math.max(acc.right, rect.right) - Math.min(acc.left, rect.left),
        height: Math.max(acc.bottom, rect.bottom) - Math.min(acc.top, rect.top)
      };
    }, { left: Infinity, top: Infinity, right: -Infinity, bottom: -Infinity });

    newElement = document.createElement("div");
    var borderColor = getRandomColor();
    newElement.style.outline = `2px dashed ${borderColor}`;
    newElement.style.position = "fixed";
    newElement.style.left = scrollableRect.left + "px";
    newElement.style.top = scrollableRect.top + "px";
    newElement.style.width = scrollableRect.width + "px";
    newElement.style.height = scrollableRect.height + "px";
    newElement.style.pointerEvents = "none";
    newElement.style.boxSizing = "border-box";
    newElement.style.zIndex = 2147483647;

    // Add floating label at the bottom right corner
    var label = document.createElement("span");
    label.textContent = index + nonScrollableItems.length; // Continue numbering
    label.style.position = "absolute";
    // Adjusted to be at the bottom right corner
    label.style.bottom = "-18px";
    label.style.right = "-20px";
    label.style.background = borderColor;
    label.style.color = "white";
    label.style.padding = "2px 4px";
    label.style.fontSize = "12px";
    label.style.borderRadius = "2px";
    label.style.opacity = "0.5"; // Make it translucent
    newElement.appendChild(label);

    document.body.appendChild(newElement);
    labels.push(newElement);
  });

  const coordinates = items.flatMap((item, index) =>
    item.rects.map(({ left, top, width, height }) => ({
      x: (left + left + width) / 2,
      y: (top + top + height) / 2,
      type: item.type,
      text: item.text,
      ariaLabel: item.ariaLabel,
      outerHTML: item.element.outerHTML,
      isScrollable: item.isScrollableArea, // Add isScrollable field
      index: index // Add index field
    }))
  );
  return coordinates;
}