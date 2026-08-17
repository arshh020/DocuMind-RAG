---
title: CSS Flexible Box Layout
slug: Web/CSS/CSS_flexible_box_layout
page-type: css-module
---

# CSS Flexible Box Layout

The CSS flexible box layout module defines a one-dimensional layout model for
distributing space between items in an interface, along with alignment
capabilities. A flex container expands items to fill available free space or
shrinks them to prevent overflow.

## Setting up a flex container

A flex container is created by setting the `display` property of an element to
`flex` or `inline-flex`. Its direct children then become flex items.

```css
.container {
  display: flex;
  gap: 1rem;
}
```

### The main axis and the cross axis

Everything in flexbox depends on two axes. The main axis is defined by
`flex-direction`, and the cross axis runs perpendicular to it. When
`flex-direction` is `row`, the main axis is horizontal and the cross axis is
vertical. When it is `column`, the two swap.

This is the single most common source of confusion: `justify-content` always
aligns along the main axis, and `align-items` always aligns along the cross
axis. Changing `flex-direction` therefore changes what those two properties
appear to do.

## Centering an element

To center a single item both horizontally and vertically, combine
`justify-content` and `align-items` on the container.

```css
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}
```

## Controlling how items grow and shrink

The `flex` shorthand sets three properties at once: `flex-grow`, `flex-shrink`,
and `flex-basis`.

```css
.item {
  flex: 1 1 auto;
}
```

`flex-grow` distributes positive free space as a proportion. Two items with
`flex-grow: 1` and `flex-grow: 2` do not end up in a 1:2 size ratio; instead the
leftover space is split in a 1:2 ratio and added to each item's base size.

`flex-basis` sets the initial size before free space is distributed. A value of
`auto` uses the item's content size or its `width`, while `0` ignores the
content size entirely.

### Preventing overflow of long content

Flex items have a default `min-width` of `auto`, which means they refuse to
shrink below their content's minimum size. A long unbroken string or a wide
child can therefore push a flex item outside its container. Setting
`min-width: 0` on the item restores shrinking.

```css
.item {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

## Wrapping onto multiple lines

By default flex items stay on a single line and shrink to fit. Setting
`flex-wrap: wrap` allows them to break onto new lines. Once a flex container is
multiline, `align-content` controls the distribution of the lines themselves,
which is a different property from `align-items`.

## Flexbox compared with grid

Flexbox lays out content in one dimension at a time and sizes items primarily
from their content. CSS grid lays out content in two dimensions on a track
grid defined in advance. Use flexbox for a row of buttons or a navigation bar,
and grid for a page skeleton or any layout where rows and columns must line up
with each other.
